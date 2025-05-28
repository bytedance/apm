"""
Copyright (2025) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0
"""


import torch
from torch import nn

from apm.models.node_feature_net import FullNodeFeatureNet, PackingNodeFeatureNet
from apm.models.edge_feature_net import FullEdgeFeatureNet
from apm.models import ipa_pytorch
from apm.analysis import utils as au
from apm.data import utils as du
import copy
import numpy as np
from torch_scatter import scatter_sum
from apm.data import so3_utils

class AngleResnetBlock(nn.Module):
    def __init__(self, c_hidden):
        """
        Args:
            c_hidden:
                Hidden channel dimension
        """
        super(AngleResnetBlock, self).__init__()

        self.c_hidden = c_hidden

        self.linear_1 = ipa_pytorch.Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_2 = ipa_pytorch.Linear(self.c_hidden, self.c_hidden, init="final")

        self.relu = nn.ReLU()

    def forward(self, a: torch.Tensor) -> torch.Tensor:

        s_initial = a

        a = self.relu(a)
        a = self.linear_1(a)
        a = self.relu(a)
        a = self.linear_2(a)

        return a + s_initial

class AngleResnet(nn.Module):
    """
    Implements Algorithm 20, lines 11-14
    """

    def __init__(self, c_in, c_hidden, no_blocks, no_angles, epsilon):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Hidden channel dimension
            no_blocks:
                Number of resnet blocks
            no_angles:
                Number of torsion angles to generate
            epsilon:
                Small constant for normalization
        """
        super(AngleResnet, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_angles = no_angles
        self.eps = epsilon

        self.linear_in = ipa_pytorch.Linear(self.c_in, self.c_hidden)
        self.linear_initial = ipa_pytorch.Linear(self.c_in, self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = AngleResnetBlock(c_hidden=self.c_hidden)
            self.layers.append(layer)

        self.linear_out = ipa_pytorch.Linear(self.c_hidden, self.no_angles * 2)

        self.relu = nn.ReLU()

    def forward(
        self, s: torch.Tensor, s_initial: torch.Tensor
    ):
        """
        Args:
            s:
                [*, C_hidden] single embedding
            s_initial:
                [*, C_hidden] single embedding as of the start of the
                StructureModule
        Returns:
            [*, no_angles, 2] predicted angles
        """
        # NOTE: The ReLU's applied to the inputs are absent from the supplement
        # pseudocode but present in the source. For maximal compatibility with
        # the pretrained weights, I'm going with the source.

        # [*, C_hidden]
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial

        for l in self.layers:
            s = l(s)

        s = self.relu(s)

        # [*, no_angles * 2]
        s = self.linear_out(s)

        # [*, no_angles, 2]
        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(s ** 2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        s = s / norm_denom

        return unnormalized_s, s

class SideChainModel(nn.Module):

    def __init__(self, model_conf, PLM_info):
        super(SideChainModel, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE)
        self.flow_based = False
        if self.flow_based:
            self.node_feature_net = FullNodeFeatureNet(model_conf.node_features)
        else:
            self.node_feature_net = PackingNodeFeatureNet(model_conf.node_features)
        self.edge_feature_net = FullEdgeFeatureNet(model_conf.edge_features)

        node_embed_size = self._model_conf.node_embed_size
        if self._model_conf.torsions_pred:
            self.torsion_pred_net = AngleResnet(c_in=node_embed_size, 
                                                c_hidden=node_embed_size, 
                                                no_blocks=self._model_conf.sidechain_model.num_torsion_blocks, 
                                                no_angles=4, 
                                                epsilon=1e-4)
            # 4: 4 side-chain torsion angles
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
        for b in range(self._model_conf.sidechain_model.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
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

            if b < self._model_conf.sidechain_model.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )


    def forward(self, input_feats):
        
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        diffuse_mask = input_feats['diffuse_mask']
        chain_index = input_feats['chain_idx']
        res_index = input_feats['res_idx']

        so3_t = input_feats['mod_so3_t'] if 'mod_so3_t' in input_feats else input_feats['so3_t']
        r3_t = input_feats['mod_r3_t'] if 'mod_r3_t' in input_feats else input_feats['r3_t']
        cat_t = input_feats['mod_cat_t'] if 'mod_cat_t' in input_feats else input_feats['cat_t']
        tor_t = input_feats['mod_tor_t'] if 'mod_tor_t' in input_feats else input_feats['tor_t']

        trans_t_pred_1 = input_feats['backbone_pred_trans'].detach().clone()
        trans_1 = input_feats['trans_1']
        fix_trans_mask = r3_t.eq(1).int().float()
        trans_for_sidechain = trans_1 * fix_trans_mask[..., None] + trans_t_pred_1 * (1 - fix_trans_mask)[..., None]

        rotmats_t_pred_1 = input_feats['backbone_pred_rotmats'].detach().clone()
        rotmats_1 = input_feats['rotmats_1']
        fix_rotmats_mask = so3_t.eq(1).int().float()
        rotmats_for_sidechain = rotmats_1 * fix_rotmats_mask[..., None, None] + rotmats_t_pred_1 * (1 - fix_rotmats_mask)[..., None, None]

        aatypes_t_pred_1 = input_feats['backbone_pred_aatypes'].detach().clone().long()
        aatypes_1 = input_feats['aatypes_1']
        fix_aatypes_mask = cat_t.eq(1).int().to(aatypes_t_pred_1.dtype)
        aatypes_for_sidechain = aatypes_1 * fix_aatypes_mask + aatypes_t_pred_1 * (1 - fix_aatypes_mask)

        if self.flow_based:
            torsions_t = input_feats['torsions_t'] # if use initilized torsion instead noised torsion_t, sidechain model will not be a flow model
            logits_t_pred_1 = input_feats['backbone_pred_logits'].detach().clone()
            logits_1 = input_feats['logits_1'] * 100.0
            logits_for_sidechain = logits_1 * fix_aatypes_mask[..., None] + logits_t_pred_1 * (1 - fix_aatypes_mask)[..., None]
        else:
            num_batch, num_res = aatypes_t_pred_1.shape[:2]
            torsions_t = torch.rand(num_batch, num_res, 4).to(aatypes_t_pred_1.device) # [0-1]
            torsions_t = torsions_t * 2 * np.pi # [0-2pi]
        torsions_sc = input_feats['torsions_sc']
        
        # Initialize node and edge embeddings
        if self.flow_based:
            init_node_embed = self.node_feature_net(
                so3_t=so3_t,
                r3_t=r3_t,
                cat_t=cat_t,
                tor_t=tor_t,
                res_mask=node_mask,
                diffuse_mask=diffuse_mask,
                chain_index=chain_index,
                pos=res_index,
                aatypes=aatypes_for_sidechain,
                aatype_logits=logits_for_sidechain,
                rotvecs=so3_utils.rotmat_to_rotvec(rotmats_for_sidechain),
                torsions=torsions_t,
                torsions_sc=torsions_sc,
            )
        else:
            # Packing model node feature nework
            init_node_embed = self.node_feature_net(
                tor_t=tor_t,
                res_mask=node_mask,
                diffuse_mask=diffuse_mask,
                chain_index=chain_index,
                pos=res_index,
                aatypes=aatypes_for_sidechain,
                rotvecs=so3_utils.rotmat_to_rotvec(rotmats_for_sidechain),
                torsions=torsions_t,
                torsions_sc=torsions_sc,
            )
        # TODO: add cat_t/so3_t/r3_t and other features to the packing model output

        if self.use_PLM:
            plm_s = (self.plm_s_combine.softmax(0).unsqueeze(0) @ input_feats['PLM_embedding_aatypes_t']).squeeze(2)
            plm_s = self.plm_s_mlp(plm_s)
            init_node_embed += plm_s
        
        # the pair-feature net in the side-chain model is actually a full-atom pair-feature net
        # thus the encoding information should contains: node_repr, bb_trans, bb_rot, sidechain_tor
        init_edge_embed = self.edge_feature_net(
            init_node_embed,
            trans_for_sidechain,
            torsions_t,
            torsions_sc, 
            edge_mask,
            diffuse_mask,
            chain_index
        )
        
        # Main trunk
        # curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        curr_rigids = du.create_rigid(rotmats_for_sidechain, trans_for_sidechain)
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]
        for b in range(self._model_conf.sidechain_model.num_blocks):
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
            
            if b < self._model_conf.sidechain_model.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        if self._model_conf.torsions_pred:
            unnorm_pred_torsion_sincos, pred_torsion_sincos = self.torsion_pred_net(node_embed, init_node_embed)
            pred_torsion = torch.atan2(pred_torsion_sincos[..., 0], pred_torsion_sincos[..., 1])
            
        return_dict = {
            'sidechain_pred_torsions': pred_torsion,
            'sidechain_pred_torsions_sincos':pred_torsion_sincos, 
            'sidechain_pred_torsions_sincos_unnorm':unnorm_pred_torsion_sincos, 
        }
        return return_dict
