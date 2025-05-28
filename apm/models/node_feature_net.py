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


import os
import numpy as np
import torch
from torch import nn
from torch.nn.functional import softmax
from apm.models.utils import get_index_embedding, get_time_embedding, sinusoidal_encoding, AngularEncoding
# import esm
# from torch_geometric import utils as tgu

class BackboneNodeFeatureNet(nn.Module):

    def __init__(self, module_cfg, PLM_N_dim=None, highlight_interface=False):
        super(BackboneNodeFeatureNet, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        self.highlight_interface = highlight_interface
        embed_size = self._cfg.c_pos_emb + self._cfg.c_timestep_emb * 2 + 1 # 385
        # if self._cfg.embed_chain:
        #     embed_size += self._cfg.c_pos_emb
        if self._cfg.embed_aatype:
            self.aatype_embedding = nn.Embedding(21, self._cfg.c_aatype_emb) # Always 21 because of 20 amino acids + 1 for unk
            embed_size += self._cfg.c_aatype_emb + self._cfg.c_timestep_emb
        
        if PLM_N_dim is None:
            self.plm_emb_aatypesc = False
            embed_size += self._cfg.aatype_pred_num_tokens # the dim of self-condition aatype embedding, if not use PLM, the dim is logits with dim=21
        else:
            self.plm_emb_aatypesc = True
            embed_size += PLM_N_dim # the dim of self-condition aatype embedding, if use PLM, the dim is the reweighted embedding of PLM 
        
        if self.highlight_interface:
            self.interface_embedding = nn.Embedding(2, self._cfg.c_s)
            embed_size += self._cfg.c_s

        if 'embed_chain_in_node_feats' in self._cfg:
            self.embed_chain_in_node_feats = self._cfg.embed_chain_in_node_feats
        else:
            self.embed_chain_in_node_feats = False
        if self.embed_chain_in_node_feats:
            embed_size += self._cfg.c_pos_emb

        if self._cfg.use_mlp:
            self.linear = nn.Sequential(
                nn.Linear(embed_size, self._cfg.c_aatype_emb),
                nn.ReLU(),
                nn.Linear(self._cfg.c_aatype_emb, self._cfg.c_aatype_emb),
                nn.ReLU(),
                nn.Linear(self._cfg.c_aatype_emb, self.c_s),
                nn.LayerNorm(self.c_s),
            )
        else:
            self.linear = nn.Linear(embed_size, self.c_s)

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.c_timestep_emb,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(
            self,
            *,
            so3_t,
            r3_t,
            cat_t,
            res_mask,
            diffuse_mask,
            chain_index,
            pos,
            aatypes,
            aatypes_sc,
            plm_model_emb_weight=None,
            interface_mask=None
        ):
        # s: [b]

        # [b, n_res, c_pos_emb]
        pos_emb = get_index_embedding(pos, self.c_pos_emb, max_len=2056)
        pos_emb = pos_emb * res_mask.unsqueeze(-1)

        # [b, n_res, c_timestep_emb]
        input_feats = [
            pos_emb,
            diffuse_mask[..., None],
            self.embed_t(so3_t, res_mask),
            self.embed_t(r3_t, res_mask)
        ]
        if self._cfg.embed_aatype:
            input_feats.append(self.aatype_embedding(aatypes))
            input_feats.append(self.embed_t(cat_t, res_mask))
            if self.plm_emb_aatypesc:
                assert plm_model_emb_weight is not None
                # if use PLM, aatype_sc is also econded by PLM by reweighting the embedding of PLM according to the predicted probability
                if aatypes_sc.dtype == torch.int64:
                    ### when self-condition is turned off, aatype_sc is the initilzed zeros with dtype int64;
                    ### to get the probability, we set the probability of <MASK> to 1 (with other aatypes' probability being 0) here
                    ### which means if self-condition is turned off, the aatype_sc all <MASK>
                    aatypes_sc_prob = torch.zeros(*diffuse_mask.shape, 21).to(aatypes_sc.device)
                    aatypes_sc_prob[:, :, -1] = 1 # the last token is <MASK>
                else:
                    ### when self-condition is turned on, aatype_sc is the predicted logits
                    aatypes_sc_prob = softmax(aatypes_sc, dim=-1) # [B, N, 21]
                # use aatype_sc to reweight embedding of PLM
                aatypes_sc_reweighted_plm_emb = aatypes_sc_prob @ plm_model_emb_weight # [B, N, esm_dim]
                input_feats.append(aatypes_sc_reweighted_plm_emb)
            else:
                input_feats.append(aatypes_sc)
        
        if self.highlight_interface:
            input_feats.append(self.interface_embedding(interface_mask))
        
        if self.embed_chain_in_node_feats:
            input_feats.append(
                get_index_embedding(
                    chain_index,
                    self.c_pos_emb,
                    max_len=36
                )
            )
        return self.linear(torch.cat(input_feats, dim=-1))

class FullNodeFeatureNet(nn.Module):

    def __init__(self, module_cfg):
        super(FullNodeFeatureNet, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        embed_size = self._cfg.c_pos_emb + self._cfg.c_timestep_emb * 2 + 1

        # if self._cfg.embed_chain:
        #     embed_size += self._cfg.c_pos_emb
            
        if self._cfg.embed_aatype:
            self.aatype_embedding = nn.Embedding(21, self.c_s) # Always 21 because of 20 amino acids + 1 for unk
            embed_size += self.c_s + self._cfg.c_timestep_emb
            embed_size += self._cfg.aatype_pred_num_tokens # for aatype_logits
            
        # if self._cfg.embed_torsion:
        self.torsion_embedding = AngularEncoding()
        embed_size += self._cfg.c_timestep_emb + self.torsion_embedding.get_out_dim(4) * 2
        # torsion_embedding dim * 2 due to self-condition

        # embed_size += 3 * 2 # considering rotvecs_t and rotvecs_sc
        embed_size += 3 * 1 # only encode rotvecs_t
            
        if self._cfg.use_mlp:
            self.linear = nn.Sequential(
                nn.Linear(embed_size, self.c_s),
                nn.ReLU(),
                nn.Linear(self.c_s, self.c_s),
                nn.ReLU(),
                nn.Linear(self.c_s, self.c_s),
                nn.LayerNorm(self.c_s),
            )
        else:
            self.linear = nn.Linear(embed_size, self.c_s)

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.c_timestep_emb,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(
            self,
            *,
            so3_t,
            r3_t,
            cat_t,
            tor_t,
            res_mask,
            diffuse_mask,
            chain_index,
            pos,
            aatypes,
            aatype_logits,
            rotvecs,
            torsions,
            torsions_sc,
        ):
        # s: [b]

        # [b, n_res, c_pos_emb]
        pos_emb = get_index_embedding(pos, self.c_pos_emb, max_len=2056)
        pos_emb = pos_emb * res_mask.unsqueeze(-1)

        # [b, n_res, c_timestep_emb]
        input_feats = [
            pos_emb,
            diffuse_mask[..., None],
            self.embed_t(so3_t, res_mask),
            self.embed_t(r3_t, res_mask), 
        ]

        if self._cfg.embed_aatype:
            input_feats.append(self.aatype_embedding(aatypes))
            input_feats.append(self.embed_t(cat_t, res_mask))
            input_feats.append(aatype_logits)

        input_feats.append(rotvecs)
        # input_feats.append(rotvecs_sc)
        
        # if self._cfg.embed_torsion:
        #     # batch_size, seq_len, _ = torsion.shape
        input_feats.append(self.embed_t(tor_t, res_mask))
        input_feats.append(self.torsion_embedding(torsions))
        input_feats.append(self.torsion_embedding(torsions_sc))
            
        # if self._cfg.embed_chain:
        #     input_feats.append(
        #         get_index_embedding(
        #             chain_index,
        #             self.c_pos_emb,
        #             max_len=100
        #         )
        #     )
            
        return self.linear(torch.cat(input_feats, dim=-1))

class PackingNodeFeatureNet(nn.Module):

    def __init__(self, module_cfg):
        super(PackingNodeFeatureNet, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        embed_size = self._cfg.c_pos_emb + 1
            
        if self._cfg.embed_aatype:
            self.aatype_embedding = nn.Embedding(21, self.c_s) # Always 21 because of 20 amino acids + 1 for unk
            embed_size += self.c_s

        self.torsion_embedding = AngularEncoding()
        embed_size += self._cfg.c_timestep_emb + self.torsion_embedding.get_out_dim(4) * 2
        # torsion_embedding dim * 2 due to self-condition

        embed_size += 3 * 1 # only encode rotvecs_1
            
        if self._cfg.use_mlp:
            self.linear = nn.Sequential(
                nn.Linear(embed_size, self.c_s),
                nn.ReLU(),
                nn.Linear(self.c_s, self.c_s),
                nn.ReLU(),
                nn.Linear(self.c_s, self.c_s),
                nn.LayerNorm(self.c_s),
            )
        else:
            self.linear = nn.Linear(embed_size, self.c_s)

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.c_timestep_emb,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(
            self,
            *,
            tor_t,
            res_mask,
            diffuse_mask,
            chain_index,
            pos,
            aatypes,
            rotvecs,
            torsions,
            torsions_sc,
        ):
        # s: [b]

        # [b, n_res, c_pos_emb]
        pos_emb = get_index_embedding(pos, self.c_pos_emb, max_len=2056)
        pos_emb = pos_emb * res_mask.unsqueeze(-1)

        # [b, n_res, c_timestep_emb]
        input_feats = [
            pos_emb,
            diffuse_mask[..., None],
        ]

        if self._cfg.embed_aatype:
            input_feats.append(self.aatype_embedding(aatypes))

        input_feats.append(rotvecs)
        
        input_feats.append(self.embed_t(tor_t, res_mask))
        input_feats.append(self.torsion_embedding(torsions))
        input_feats.append(self.torsion_embedding(torsions_sc))
            
        return self.linear(torch.cat(input_feats, dim=-1))

class RefineNodeFeatureNet(nn.Module):

    def __init__(self, module_cfg):
        super(RefineNodeFeatureNet, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        embed_size = self._cfg.c_pos_emb + self._cfg.c_timestep_emb * 2 + 1

        if 'embed_chain_in_node_feats' in self._cfg:
            self.embed_chain_in_node_feats = self._cfg.embed_chain_in_node_feats
        else:
            self.embed_chain_in_node_feats = False

        if self.embed_chain_in_node_feats:
            embed_size += self._cfg.c_pos_emb
            
        if self._cfg.embed_aatype:
            self.aatype_embedding = nn.Embedding(21, self.c_s) # Always 21 because of 20 amino acids + 1 for unk
            embed_size += self.c_s + self._cfg.c_timestep_emb
            encode_aatype_sc = False
            if encode_aatype_sc:
                embed_size += self._cfg.aatype_pred_num_tokens
            
        # if self._cfg.embed_torsion:
        self.torsion_embedding = AngularEncoding()
        # embed_size += self._cfg.c_timestep_emb + self.torsion_embedding.get_out_dim(4) * 1 
        # *1 because we do not consider self-conditioning here
        # (x) torsion_embedding dim * 2 due to self-condition
        embed_size += self.torsion_embedding.get_out_dim(4) * 1 # as we don't use flow-based packing model, we don not encode tor_t here

        embed_size += 3 # for rotvecs
        # embed_size += 3 * 2 # considering rotvecs_t and x (rotvecs_sc)
            
        if self._cfg.use_mlp:
            self.linear = nn.Sequential(
                nn.Linear(embed_size, self.c_s),
                nn.ReLU(),
                nn.Linear(self.c_s, self.c_s),
                nn.ReLU(),
                nn.Linear(self.c_s, self.c_s),
                nn.LayerNorm(self.c_s),
            )
        else:
            self.linear = nn.Linear(embed_size, self.c_s)

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.c_timestep_emb,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(
            self,
            *,
            so3_t,
            r3_t,
            cat_t,
            tor_t,
            res_mask,
            diffuse_mask,
            chain_index,
            pos,
            aatypes,
            rotvecs,
            torsions,
            # aatypes_sc,
            # rotvecs_sc,
        ):
        # s: [b]

        # [b, n_res, c_pos_emb]
        pos_emb = get_index_embedding(pos, self.c_pos_emb, max_len=2056)
        pos_emb = pos_emb * res_mask.unsqueeze(-1)

        # [b, n_res, c_timestep_emb]
        input_feats = [
            pos_emb,
            diffuse_mask[..., None],
            self.embed_t(so3_t, res_mask),
            self.embed_t(r3_t, res_mask), 
        ]

        if self._cfg.embed_aatype:
            input_feats.append(self.aatype_embedding(aatypes))
            input_feats.append(self.embed_t(cat_t, res_mask))
            # input_feats.append(aatypes_sc)

        input_feats.append(rotvecs)
        
        # input_feats.append(self.embed_t(tor_t, res_mask))
        input_feats.append(self.torsion_embedding(torsions))
            
        if self.embed_chain_in_node_feats:
            input_feats.append(
                get_index_embedding(
                    chain_index,
                    self.c_pos_emb,
                    max_len=36
                )
            )
            
        return self.linear(torch.cat(input_feats, dim=-1))
