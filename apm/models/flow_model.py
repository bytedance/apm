"""
----------------
MIT License

Copyright (c) 2024 Andrew Campbell, Jason Yim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
----------------
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”). 
All Bytedance's Modifications are Copyright (2025) Bytedance Ltd. and/or its affiliates. 
"""


import torch
from torch import nn

from apm.models.node_feature_net import BackboneNodeFeatureNet as NodeFeatureNet
from apm.models.edge_feature_net import EdgeFeatureNet
from apm.models import ipa_pytorch
from apm.analysis import utils as au
from apm.data import utils as du
# from apm.data.foldflow.rot_operator import vectorfield

class BackboneModel(nn.Module):

    def __init__(self, model_conf, PLM_info):
        super(BackboneModel, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE)
        self.use_PLM = PLM_info[0] is not None
        self.residual_PLM_embed = self._model_conf.residual_PLM_embed
        self.highlight_interface = self._model_conf.highlight_interface if 'highlight_interface' in self._model_conf else False
        self.node_feature_net = NodeFeatureNet(model_conf.node_features, PLM_N_dim=PLM_info[1], highlight_interface=self.highlight_interface)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features, highlight_interface=self.highlight_interface)

        if self._model_conf.aatype_pred:
            node_embed_size = self._model_conf.node_embed_size
            self.aatype_pred_net = nn.Sequential(
                nn.Linear(node_embed_size, node_embed_size),
                nn.ReLU(),
                nn.Linear(node_embed_size, node_embed_size),
                nn.ReLU(),
                nn.Linear(node_embed_size, self._model_conf.aatype_pred_num_tokens),
            )

        """
        ### use PLM in EsmFold stype
        """
        if self.use_PLM:
            self.plm_s_combine = nn.Parameter(torch.zeros(PLM_info[0] + 1))
            self.plm_s_mlp = nn.Sequential(
                nn.LayerNorm(PLM_info[1]),
                nn.Linear(PLM_info[1], self._model_conf.node_embed_size),
                nn.ReLU(),
                nn.Linear(self._model_conf.node_embed_size, self._model_conf.node_embed_size),
            ) # self._model_conf.node_embed_size == self._model_conf.*.c_s

            if self._model_conf.use_plm_attn_map:
                self.plm_attn_dim = 20 * PLM_info[0]
                self.plm_z_mlp = nn.Sequential(
                    nn.LayerNorm(self.plm_attn_dim),
                    nn.Linear(self.plm_attn_dim, self._model_conf.edge_embed_size),
                    nn.ReLU(),
                    nn.Linear(self._model_conf.edge_embed_size, self._model_conf.edge_embed_size),
                ) # self._model_conf.edge_embed_size == self._model_conf.*.c_s
            
            if self.residual_PLM_embed:
                self.plm_s_combine_post = nn.Parameter(torch.zeros(PLM_info[0] + 1))
                self.plm_s_mlp_post = nn.Sequential(
                    nn.LayerNorm(PLM_info[1]),
                    nn.Linear(PLM_info[1], self._model_conf.node_embed_size),
                    nn.ReLU(),
                    nn.Linear(self._model_conf.node_embed_size, self._model_conf.node_embed_size),
                )
                self.aatype_pre_pred_ln = nn.LayerNorm(self._ipa_conf.c_s)

        # Attention trunk
        self.trunk = nn.ModuleDict()

        self.pre_trunk_ln_s = nn.LayerNorm(self._ipa_conf.c_s)
        self.pre_trunk_ln_z = nn.LayerNorm(self._ipa_conf.c_z)
        self.linear_in = ipa_pytorch.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s)
        self.ipa_dropout = nn.Dropout(self._ipa_conf.dropout)

        for b in range(self._model_conf.backbone_model.num_blocks):
            # self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)

            tfmr_in = self._ipa_conf.c_output

            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            # self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttentionMultimer(self._ipa_conf, output_dim=tfmr_in)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            
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

            if b < self._model_conf.backbone_model.num_blocks - 1:
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

        trans_t = input_feats['trans_t']
        rotmats_t = input_feats['rotmats_t']
        aatypes_t = input_feats['aatypes_t'].long()
        trans_sc = input_feats['trans_sc']
        aatypes_sc = input_feats['aatypes_sc']
        interface_mask = input_feats['interface_mask'] if 'interface_mask' in input_feats else None

        # Initialize node and edge embeddings
        init_node_embed = self.node_feature_net(
            so3_t=so3_t,
            r3_t=r3_t,
            cat_t=cat_t,
            res_mask=node_mask,
            diffuse_mask=diffuse_mask,
            chain_index=chain_index,
            pos=res_index,
            aatypes=aatypes_t,
            aatypes_sc=aatypes_sc,
            plm_model_emb_weight=input_feats['PLM_emb_weight'],
            interface_mask=interface_mask,
        )

        if self.use_PLM:
            plm_s = (self.plm_s_combine.softmax(0).unsqueeze(0) @ input_feats['PLM_embedding_aatypes_t']).squeeze(2)
            plm_s = self.plm_s_mlp(plm_s)
            init_node_embed += plm_s

        init_edge_embed = self.edge_feature_net(
            init_node_embed,
            trans_t,
            trans_sc,
            edge_mask,
            diffuse_mask,
            chain_index,
            interface_mask=interface_mask,
        )

        if self.use_PLM and self._model_conf.use_plm_attn_map:
            plm_z = plm_z.to(init_edge_embed.dtype)
            plm_z = plm_z.detach()
            plm_z = self.plm_z_mlp(plm_z)
            init_edge_embed += plm_z

        # Initial rigids
        # init_rigids = du.create_rigid(rotmats_t, trans_t)
        curr_rigids = du.create_rigid(rotmats_t, trans_t)

        # Main trunk
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]
        node_embed = self.pre_trunk_ln_s(node_embed)
        edge_embed = self.pre_trunk_ln_z(edge_embed)
        node_embed = self.linear_in(node_embed)
        for b in range(self._model_conf.backbone_model.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            # node_embed = self.ipa_dropout(node_embed + ipa_embed)
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool))
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, (node_mask * diffuse_mask)[..., None])
            if b < self._model_conf.backbone_model.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # _, rot_vectorfield = vectorfield(
        #     curr_rigids.get_rots().get_rot_mats(),
        #     init_rigids.get_rots().get_rot_mats(),
        #     so3_t,
        # )

        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()
        if self._model_conf.aatype_pred:
            if self.residual_PLM_embed:
                plm_s_post = (self.plm_s_combine_post.softmax(0).unsqueeze(0) @ input_feats['PLM_embedding_aatypes_t']).squeeze(2)
                plm_s_post = self.plm_s_mlp(plm_s_post)
                node_embed += plm_s_post
                node_embed = self.aatype_pre_pred_ln(node_embed)
            pred_logits = self.aatype_pred_net(node_embed)
            pred_aatypes = torch.argmax(pred_logits, dim=-1)
            if self._model_conf.aatype_pred_num_tokens == du.NUM_TOKENS + 1:
                pred_logits_wo_mask = pred_logits.clone()
                pred_logits_wo_mask[:, :, du.MASK_TOKEN_INDEX] = -1e9
                pred_aatypes = torch.argmax(pred_logits_wo_mask, dim=-1)
            else:
                pred_aatypes = torch.argmax(pred_logits, dim=-1)
        else:
            pred_aatypes = aatypes_t
            pred_logits = nn.functional.one_hot(
                pred_aatypes, num_classes=self._model_conf.aatype_pred_num_tokens
            ).float()
        return {
            'backbone_pred_trans': pred_trans,
            'backbone_pred_rotmats': pred_rotmats,
            'backbone_pred_logits': pred_logits,
            'backbone_pred_aatypes': pred_aatypes,
            'backbone_init_node_embed': init_node_embed.detach(),
            'backbone_init_edge_embed': init_edge_embed.detach(),
        }
