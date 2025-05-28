"""
Copyright (2025) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0
"""


import torch
from torch import nn

from apm.models.node_feature_net import RefineNodeFeatureNet
from apm.models.edge_feature_net import RefineEdgeFeatureNet
from apm.models import ipa_pytorch
from apm.analysis import utils as au
from apm.data import utils as du

from torch_scatter import scatter_sum
from apm.data import so3_utils
import copy


class GateUpdate(nn.Module):
    def __init__(self, dim=6):
        super(GateUpdate, self).__init__()
        self.gate = nn.Parameter(torch.zeros(dim,))
    
    def forward(self, update):
        return update * self.gate[None, None, :]# .tanh()

class RefineModel(nn.Module):

    def __init__(self, model_conf, PLM_info):
        super(RefineModel, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE) 
        self.node_feature_net = RefineNodeFeatureNet(model_conf.node_features)
        self.edge_feature_net = RefineEdgeFeatureNet(model_conf.edge_features)

        if self._model_conf.aatype_pred:
            node_embed_size = self._model_conf.node_embed_size
            self.aatype_pred_net = nn.Sequential(
                ipa_pytorch.Linear(node_embed_size, node_embed_size),
                nn.ReLU(),
                ipa_pytorch.Linear(node_embed_size, node_embed_size),
                nn.ReLU(),
                ipa_pytorch.Linear(node_embed_size, self._model_conf.aatype_pred_num_tokens, init="final"),
            )
        
        self.use_PLM = PLM_info[0] is not None
        if self.use_PLM:
            self.plm_s_combine = nn.Parameter(torch.zeros(PLM_info[0] + 1))
            self.plm_s_mlp = nn.Sequential(
                nn.LayerNorm(PLM_info[1]),
                ipa_pytorch.Linear(PLM_info[1], self._model_conf.node_embed_size, init='relu'),
                nn.ReLU(),
                ipa_pytorch.Linear(self._model_conf.node_embed_size, self._model_conf.node_embed_size, init='final'),
            )
        
        # Attention trunk
        self.trunk = nn.ModuleDict()

        self.pre_trunk_ln_s = nn.LayerNorm(self._ipa_conf.c_s)
        self.pre_trunk_ln_z = nn.LayerNorm(self._ipa_conf.c_z)
        self.linear_in = ipa_pytorch.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s)

        for b in range(self._model_conf.refine_model.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=2*tfmr_in,
                batch_first=True,
                dropout=self._model_conf.transformer_dropout,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)
            # self.trunk[f'bb_update_{b}_gate'] = GateUpdate(6)

            if b < self._model_conf.refine_model.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )
                
        
        # self.gate_aatype = GateUpdate(21)



    def forward(self, input_feats):
        
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        diffuse_mask = input_feats['diffuse_mask']
        chain_index = input_feats['chain_idx']
        res_index = input_feats['res_idx']
        so3_t = input_feats['mod_so3_t'] if 'mod_so3_t' in input_feats else input_feats['so3_t']
        r3_t = input_feats['mod_r3_t'] if 'mod_r3_t' in input_feats else input_feats['r3_t']
        cat_t = input_feats['mod_cat_t'] if 'mod_cat_t' in input_feats else input_feats['cat_t']
        # tor_t = input_feats['mod_tor_t'] if 'mod_tor_t' in input_feats else input_feats['tor_t']
        
        trans_t_pred_1 = input_feats['backbone_pred_trans'].detach().clone()
        trans_1 = input_feats['trans_1']
        fix_trans_mask = r3_t.eq(1).int().float()
        trans_for_refine = trans_1 * fix_trans_mask[..., None] + trans_t_pred_1 * (1 - fix_trans_mask)[..., None]

        rotmats_t_pred_1 = input_feats['backbone_pred_rotmats'].detach().clone()
        rotmats_1 = input_feats['rotmats_1']
        fix_rotmats_mask = so3_t.eq(1).int().float()
        rotmats_for_refine = rotmats_1 * fix_rotmats_mask[..., None, None] + rotmats_t_pred_1 * (1 - fix_rotmats_mask)[..., None, None]

        logits_t_pred_1 = input_feats['backbone_pred_logits'].detach().clone()
        logits_1 = input_feats['logits_1'] * 100.0
        aatypes_t_pred_1 = input_feats['backbone_pred_aatypes'].detach().clone().long()
        aatypes_1 = input_feats['aatypes_1']
        fix_aatypes_mask = cat_t.eq(1).int().to(aatypes_t_pred_1.dtype)
        logits_for_refine = logits_1 * fix_aatypes_mask[..., None] + logits_t_pred_1 * (1 - fix_aatypes_mask)[..., None]
        aatypes_for_refine = aatypes_1 * fix_aatypes_mask + aatypes_t_pred_1 * (1 - fix_aatypes_mask)

        torsions_for_refine = input_feats['sidechain_pred_torsions'].detach().clone()
        
        # Initialize node and edge embeddings
        init_node_embed = self.node_feature_net(
            so3_t=so3_t,
            r3_t=r3_t,
            cat_t=cat_t,
            tor_t=None,
            res_mask=node_mask,
            diffuse_mask=diffuse_mask,
            chain_index=chain_index,
            pos=res_index,
            aatypes=aatypes_for_refine,
            rotvecs=so3_utils.rotmat_to_rotvec(rotmats_for_refine),
            torsions=torsions_for_refine,
        )
        
        if self.use_PLM:
            plm_s = (self.plm_s_combine.softmax(0).unsqueeze(0) @ input_feats['PLM_embedding_aatypes_t']).squeeze(2)
            plm_s = self.plm_s_mlp(plm_s)
            init_node_embed += plm_s

        init_edge_embed = self.edge_feature_net(
            init_node_embed,
            trans_for_refine,
            torsions_for_refine, 
            edge_mask,
            diffuse_mask,
            chain_index
        )
        
        # Main trunk
        curr_rigids = du.create_rigid(rotmats_for_refine, trans_for_refine)
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]
        node_embed = self.pre_trunk_ln_s(node_embed)
        edge_embed = self.pre_trunk_ln_z(edge_embed)
        node_embed = self.linear_in(node_embed)
        for b in range(self._model_conf.refine_model.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool))
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            # rigid_update = self.trunk[f'bb_update_{b}_gate'](rigid_update)
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, (node_mask * diffuse_mask)[..., None])
            if b < self._model_conf.refine_model.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        refined_trans = curr_rigids.get_trans()
        refined_rotmats = curr_rigids.get_rots().get_rot_mats()
        
        if self._model_conf.aatype_pred:
            delta_logits = self.aatype_pred_net(node_embed)
            # refined_logits = logits_for_refine + self.gate_aatype(delta_logits)
            refined_logits = logits_for_refine + delta_logits
            refined_aatypes = torch.argmax(refined_logits, dim=-1)
            if self._model_conf.aatype_pred_num_tokens == du.NUM_TOKENS + 1:
                refined_logits_wo_mask = refined_logits.clone()
                refined_logits_wo_mask[:, :, du.MASK_TOKEN_INDEX] = -1e9
                refined_aatypes = torch.argmax(refined_logits_wo_mask, dim=-1)
            else:
                refined_aatypes = torch.argmax(refined_logits, dim=-1)
        else:
            refined_aatypes = aatypes_t_pred_1
            refined_logits = logits_for_refine
            
        return_dict = {
            'refine_pred_trans': refined_trans,
            'refine_pred_rotmats': refined_rotmats,
            'refine_pred_aatypes': refined_aatypes,
            'refine_pred_logits': refined_logits,
            'refine_pred_rigids': curr_rigids
        }
        return return_dict
