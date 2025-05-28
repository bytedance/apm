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


from typing import Any
import torch
from torch import nn
import time
import os
import random
import wandb
import numpy as np
import pandas as pd
import logging
import shutil
from einops import rearrange
import torch.distributed as dist
from glob import glob
import copy
from pytorch_lightning import LightningModule
from apm.analysis import utils as au
from apm.models.flow_model import BackboneModel
from apm.models.side_chain_model import SideChainModel
from apm.models.refine_model import RefineModel
from apm.models import utils as mu
from apm.models import folding_model
from apm.data.interpolant import Interpolant, _centered_gaussian, _uniform_torsion, _masked_categorical
from apm.data import utils as du
from apm.data import all_atom, so3_utils, so2_utils
from apm.data.residue_constants import restypes, restypes_with_x
from apm.data import residue_constants
from apm.experiments import utils as eu
from biotite.sequence.io import fasta
from pytorch_lightning.loggers.wandb import WandbLogger

from openfold.utils.loss import backbone_loss, supervised_chi_loss, sidechain_loss, compute_renamed_ground_truth
from openfold.utils.rigid_utils import Rotation, Rigid
from openfold.np.residue_constants import restype_rigid_group_default_frame, restype_atom14_to_rigid_group, restype_atom14_mask, restype_atom14_rigid_group_positions
from openfold.utils.feats import frames_and_literature_positions_to_atom14_pos, torsion_angles_to_frames

# from apm.data.foldflow.rot_operator import foldflow_rotmat_to_vf, foldflow_rot_loss
from torch.utils.data.distributed import dist

class FlowModule(LightningModule):

    def __init__(self, cfg, dataset_cfg, folding_cfg=None, folding_device_id=None):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        self._dataset_cfg = dataset_cfg
        self._interpolant_cfg = cfg.interpolant

        # Set-up vector field prediction model
        if cfg.interpolant.aatypes.interpolant_type == 'uniform':
            cfg.model.aatype_pred_num_tokens=20

        # Set-up interpolant
        self.interpolant = Interpolant(cfg.interpolant)

        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
        self.save_hyperparameters()

        self._checkpoint_dir = None
        self._inference_dir = None

        # self._folding_model = None
        self._folding_cfg = folding_cfg
        self._folding_device_id = folding_device_id
        self.folding_model = folding_model.FoldingModel(
                self._folding_cfg,
                device_id=self._folding_device_id
            )
        
        if self._folding_cfg.PLM is not None:
            self.use_PLM = self._folding_cfg.PLM
            self.PLM_info = (self.folding_model.plm_representations_layer, self.folding_model.plm_representations_dim)
        else:
            self.use_PLM = False
            self.PLM_info = (None, None)

        self.design_level = cfg.experiment.design_level
        self.model = nn.ModuleDict()
        self.model_order = []
        for model_type in ('backbone', 'sidechain', 'refine'):
            if model_type == 'backbone':
                self.model['backbone'] = BackboneModel(cfg.model, self.PLM_info)
                self.model_order.append('backbone')
            elif model_type == 'sidechain':
                if cfg.packing_model.use_plm:
                    self.model['sidechain'] = SideChainModel(cfg.packing_model, self.PLM_info)
                else:
                    self.model['sidechain'] = SideChainModel(cfg.packing_model, [None, None])
                self.model_order.append('sidechain')
            else:
                self.model['refine'] = RefineModel(cfg.model, self.PLM_info)
                self.model_order.append('refine')
            if model_type == self.design_level:
                break
        self.curr_training_step = 1
        self.model_training_steps = {}
        self.model_training_orders = []
        for block_training_cfg in self._exp_cfg.training.model_training_steps.split('-'):
            block_training_cfg_ = block_training_cfg.split('_')
            model_types = block_training_cfg_[:-1]
            for model_type in model_types:
                assert model_type in self.model_order
            training_steps = int(block_training_cfg_[-1])
            if training_steps > 0:
                training_models_comb = ','.join(model_types)
                self.model_training_orders.append(training_models_comb)
                self.model_training_steps[training_models_comb] = training_steps
        self.training_loop_steps = sum([self.model_training_steps[model_comb] for model_comb in self.model_training_orders])
        
        training_schedule_str = []
        for training_models_comb in self.model_training_orders:
            training_schedule_str.append(training_models_comb)
            training_models_comb_steps = self.model_training_steps[training_models_comb]
            training_schedule_str.append(f'* {training_models_comb_steps} step')
            if training_models_comb_steps > 1:
                training_schedule_str[-1] += 's'
            training_schedule_str.append('->')
        training_schedule_str = ' '.join(training_schedule_str + [training_schedule_str[0], training_schedule_str[1], '->', '...'])
        self._print_logger.info(f'Training schedule: {training_schedule_str}')

        self.model_losses = {'backbone':['backbone_aatypes', 'backbone_trans', 'backbone_rotmats'],
                             'sidechain':['sidechain_torsions', ],
                             'refine':['refine_aatypes', 'refine_trans', 'refine_rotmats']}

        self.total_t = ('cat_t', 'r3_t', 'so3_t', 'tor_t')
        
        self.model_start_t = {'backbone':self._interpolant_cfg.min_t, 
                              'sidechain':self._interpolant_cfg.sidechain_start_t, 
                              'refine':self._interpolant_cfg.refine_start_t}

        self.d_t = float(1)/self._interpolant_cfg.sampling.num_timesteps

        self.aatype_pred_num_tokens = cfg.model.aatype_pred_num_tokens

        self.consistency_mode = self._exp_cfg.consistency_loss_weight > 0

        self.seed_has_been_set = False

        self.init_residue_constants = False

    def default_tempalte(self, term, float_dtype, device):
        if not self.init_residue_constants:
            self.residue_constants = {}
            self.default_frames = torch.tensor(restype_rigid_group_default_frame,
                                               dtype=float_dtype,
                                               device=device, 
                                               requires_grad=False,)
            self.residue_constants['default_frames'] = self.default_frames

            self.group_idx = torch.tensor(restype_atom14_to_rigid_group,
                                          device=device,
                                          requires_grad=False,)
            self.residue_constants['group_idx'] = self.group_idx

            self.atom_mask = torch.tensor(restype_atom14_mask,
                                          dtype=torch.long,
                                          device=device,
                                          requires_grad=False,)
            self.residue_constants['atom_mask'] = self.atom_mask

            self.lit_positions = torch.tensor(restype_atom14_rigid_group_positions,
                                              dtype=float_dtype,
                                              device=device,
                                              requires_grad=False,)
            self.residue_constants['lit_positions'] = self.lit_positions

            self.init_residue_constants = True
        
        return self.residue_constants[term]

    @property
    def checkpoint_dir(self):
        if self._checkpoint_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    checkpoint_dir = [self._exp_cfg.checkpointer.dirpath]
                else:
                    checkpoint_dir = [None]
                dist.broadcast_object_list(checkpoint_dir, src=0)
                checkpoint_dir = checkpoint_dir[0]
            else:
                checkpoint_dir = self._exp_cfg.checkpointer.dirpath
            self._checkpoint_dir = checkpoint_dir
            os.makedirs(self._checkpoint_dir, exist_ok=True)
        return self._checkpoint_dir

    @property
    def inference_dir(self):
        if self._inference_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    inference_dir = [self._exp_cfg.inference_dir]
                else:
                    inference_dir = [None]
                dist.broadcast_object_list(inference_dir, src=0)
                inference_dir = inference_dir[0]
            else:
                inference_dir = self._exp_cfg.inference_dir
            self._inference_dir = inference_dir
            os.makedirs(self._inference_dir, exist_ok=True)
        return self._inference_dir

    def on_train_start(self):
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'train/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )
        self._epoch_start_time = time.time()

    def on_validation_epoch_start(self):
        validation_epoch_dir = os.path.join(self.checkpoint_dir, f"epoch_{self.current_epoch}")
        if self.trainer.is_global_zero:
            print("in node 0, accelerator 0")
            os.makedirs(validation_epoch_dir, exist_ok=True)

    def run_model(self, noisy_batch, until=None, mod_t=False):

        if until is None:
            until = self.model_order

        pass_models = sorted(until, key=lambda x:{'backbone':0, 'sidechain':1, 'refine':2}[x])
        end_model = pass_models[-1]
        models_to_run = []
        for model_type in self.model_order:
            models_to_run.append(model_type)
            if model_type == end_model:
                break

        for model_type in models_to_run:
            if mod_t:
                # keeping t in any calculation within a valid range
                for t in self.total_t:
                    if t not in noisy_batch:
                        continue
                    actual_cal_t = self.interpolant.modify_t(noisy_batch[t], self.d_t, self.model_start_t[model_type])
                    noisy_batch['mod_'+t] = actual_cal_t
            
            model_output = self.model[model_type](noisy_batch)
            for k in model_output:
                noisy_batch[k] = model_output[k]
            
        return noisy_batch

    def cal_trans_loss(self, gt_trans_1, pred_trans_1, norm_scale, loss_mask):
        loss_denom = torch.sum(loss_mask, dim=-1) * 3 + 1e-4
        trans_error = (gt_trans_1 - pred_trans_1) / norm_scale * self._exp_cfg.training.trans_scale
        trans_loss = self._exp_cfg.training.translation_loss_weight * torch.sum(
            trans_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom
        trans_loss = torch.clamp(trans_loss, max=5 * self._exp_cfg.training.translation_loss_weight)
        return trans_loss
    
    def cal_rot_loss(self, gt_rotmats_1, pred_rotmats_1, rotmats_t, norm_scale, loss_mask, so3_t):
        loss_denom = torch.sum(loss_mask, dim=-1) * 3 + 1e-4
        if self._exp_cfg.rot_training_style == 'multiflow':
            gt_rot_vf = so3_utils.calc_rot_vf(
                rotmats_t, gt_rotmats_1.type(torch.float32))
            pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)
            if torch.any(torch.isnan(pred_rots_vf)):
                raise ValueError('NaN encountered in pred_rots_vf')
            rots_vf_error = (gt_rot_vf - pred_rots_vf) / norm_scale
            rots_vf_loss = self._exp_cfg.training.rotation_loss_weights * torch.sum(
                rots_vf_error ** 2 * loss_mask[..., None],
                dim=(-1, -2)
            ) / loss_denom
        elif self._exp_cfg.rot_training_style == 'foldflow':
            so3_t_foldflow_style = so3_t.reshape(-1)
            norm_scale = torch.ones_like(so3_t_foldflow_style)[:, None, None]
            try:
                gt_rot_u_t = foldflow_rotmat_to_vf(gt_rotmats_1, rotmats_t, 1-so3_t)
                pred_rot_v_t = foldflow_rotmat_to_vf(pred_rotmats_1, rotmats_t, 1-so3_t)
            except ValueError as e:
                t_shape = rotmats_t.shape[0]
                rotmats_t = rearrange(rotmats_t, "t n c d -> (t n) c d", c=3, d=3).double()
                gt_rot_u_t = torch.zeros_like(rotmats_t[..., 0])
                pred_rot_v_t = torch.zeros_like(rotmats_t[..., 0])
                gt_rot_u_t = rearrange(gt_rot_u_t, "(t n) c -> t n c", t=t_shape, c=3)
                pred_rot_v_t = rearrange(pred_rot_v_t, "(t n) c -> t n c", t=t_shape, c=3)

            rots_vf_loss = foldflow_rot_loss(gt_rot_u_t=gt_rot_u_t, 
                                             pred_rot_v_t=pred_rot_v_t, 
                                             t = 1-so3_t_foldflow_style, 
                                             loss_mask=loss_mask, 
                                             rot_vectorfield_scaling=norm_scale, 
                                             rot_loss_weight=0.5, 
                                             rot_loss_t_threshold=0.0)
            rots_vf_loss = torch.clamp(rots_vf_loss, max=5)

        else:
            raise ValueError(f'{self._exp_cfg.rot_training_style} has not been implemented')
        return rots_vf_loss
    
    def cal_aatype_loss(self, gt_aatypes_1, pred_logits, norm_scale, loss_mask):
        loss_denom = torch.sum(loss_mask, dim=-1) + 1e-4
        num_batch, num_res = loss_mask.shape
        ce_loss = torch.nn.functional.cross_entropy(
            pred_logits.reshape(-1, self.aatype_pred_num_tokens),
            gt_aatypes_1.flatten().long(),
            reduction='none',
        ).reshape(num_batch, num_res) / norm_scale
        aatypes_loss = torch.sum(ce_loss * loss_mask, dim=-1) / loss_denom
        aatypes_loss *= self._exp_cfg.training.aatypes_loss_weight
        return aatypes_loss

    def cal_consistency_loss(self, model_output, model_output_adj_t, loss_mask):
        # model_output = model(x_t, t)
        # model_output_adj_t = model(x_t+dt, t+dt), which is more close to the ground truth
        # we set model_output_adj_t as the teacher model, and the grad is stopped in model_output_adj_t
        # consistency_loss from IMPROVED TECHNIQUES FOR TRAINING CONSISTENCY MODELS: d(x,y) = sqrt((L2(x-y))^2+c^2)-c, c=0.00054*sqrt(d)

        consistency_loss = {}
        ### consistency loss for trans
        loss_mask_trans = loss_mask * model_output['r3_t'].lt(1) * model_output_adj_t['r3_t'].lt(1)
        loss_denom_trans = torch.sum(loss_mask_trans, dim=-1) * 3 + 1e-4
        consistency_trans_error = model_output['backbone_pred_trans'] - model_output_adj_t['backbone_pred_trans'] # [B, L, 3]
        consistency_trans_loss = torch.sum(consistency_trans_error ** 2, dim=-1) # [B, L], (L2(x-y))^2
        # trans_c = 0.00054 * torch.sqrt(torch.ones_like(consistency_trans_loss) * 3)
        trans_c = 0.00054 * np.sqrt(3)
        consistency_trans_loss = consistency_trans_loss + trans_c ** 2 # [B, L],  (L2(x-y))^2+c^2
        consistency_trans_loss = torch.sqrt(consistency_trans_loss) - trans_c # [B, L], sqrt((L2(x-y))^2+c^2)-c
        consistency_trans_loss = consistency_trans_loss * loss_mask_trans # [B, L]
        r3_t_reweight = model_output['r3_t'] ** 2
        consistency_trans_loss = self._exp_cfg.training.translation_loss_weight * torch.sum(
            consistency_trans_loss * r3_t_reweight, # consistenc loss will be reweighted by t
            dim=-1
        ) / loss_denom_trans # [B, ]
        consistency_trans_loss = torch.clamp(consistency_trans_loss, max=5 * self._exp_cfg.training.translation_loss_weight)
        consistency_loss['backbone_trans_consistency'] = consistency_trans_loss

        ### consistency loss for rot
        loss_mask_rot = loss_mask * model_output['so3_t'].lt(1) * model_output_adj_t['so3_t'].lt(1)
        loss_denom_rot = torch.sum(loss_mask_rot, dim=-1) * 3 + 1e-4
        consistency_rot_error = so3_utils.rotmat_to_rotvec(model_output['backbone_pred_rotmats']) - so3_utils.rotmat_to_rotvec(model_output_adj_t['backbone_pred_rotmats']) # [B, L, 3]
        consistency_rot_loss = torch.sum(consistency_rot_error ** 2, dim=-1) # [B, L], (L2(x-y))^2
        # rot_c = 0.00054 * torch.sqrt(torch.ones_like(consistency_rot_loss) * 3)
        rot_c = 0.00054 * np.sqrt(3)
        consistency_rot_loss = consistency_rot_loss + rot_c ** 2 # [B, L], (L2(x-y))^2+c^2
        consistency_rot_loss = torch.sqrt(consistency_rot_loss) - rot_c # [B, L], sqrt((L2(x-y))^2+c^2)-c
        consistency_rot_loss = consistency_rot_loss * loss_mask_rot # [B, L]
        so3_t_reweight = model_output['so3_t'] ** 2
        consistency_rot_loss = self._exp_cfg.training.rotation_loss_weights * torch.sum(
            consistency_rot_loss * so3_t_reweight,
            dim=-1
        ) / loss_denom_rot
        consistency_loss['backbone_rotvecs_consistency'] = consistency_rot_loss

        ### consistency loss for aatype
        loss_mask_aatype = loss_mask * model_output['cat_t'].lt(1) * model_output_adj_t['cat_t'].lt(1)
        loss_denom_aatype = torch.sum(loss_mask_aatype, dim=-1) + 1e-4
        model_output_aatype_logprob = torch.nn.functional.log_softmax(model_output['backbone_pred_logits'][...,:-1], dim=-1)
        model_output_adj_t_aatype_logprob = torch.nn.functional.log_softmax(model_output_adj_t['backbone_pred_logits'][...,:-1], dim=-1)
        consistency_logits_KLdiv = torch.nn.functional.kl_div(
            model_output_aatype_logprob, model_output_adj_t_aatype_logprob, reduction='none', log_target=True) # [B, L, 20]
        # consistency_logits_error = model_output['backbone_pred_logits'][...,:-1] - model_output_adj_t['backbone_pred_logits'][...,:-1] # [B, L, 20], the last dim, [MASK], is ignored
        consistency_logits_loss = torch.sum(consistency_logits_KLdiv, dim=-1) # [B, L], (L2(x-y))^2
        # cat_c = 0.00054 * torch.sqrt(torch.ones_like(consistency_logits_loss) * 20)
        cat_c = 0.00054 * np.sqrt(20)
        consistency_logits_loss = consistency_logits_loss ** 2 + cat_c ** 2 # [B, L], (L2(x-y))^2+c^2
        consistency_logits_loss = torch.sqrt(consistency_logits_loss) - cat_c # [B, L], sqrt((L2(x-y))^2+c^2)-c
        consistency_logits_loss = consistency_logits_loss * loss_mask_aatype # [B, L]
        cat_t_reweight = model_output['cat_t'] ** 2
        consistency_aatype_loss = self._exp_cfg.training.aatypes_loss_weight * torch.sum(
            consistency_logits_loss * cat_t_reweight, 
            dim=-1
        ) / loss_denom_aatype
        consistency_loss['backbone_aatype_consistency'] = consistency_aatype_loss

        return consistency_loss
    
    def cal_tor_loss(self, gt_torsions_1, pred_torsions_1, norm_scale, loss_mask):
        # gt_torsions_1: [B, N, 4]
        loss_denom = torch.sum(loss_mask, dim=-1) * 4 + 1e-4
        # torsions_vf_error = so2_utils.wrap(gt_torsions_1 - pred_torsions_1) / norm_scale
        torsions_vf_error = (gt_torsions_1 - pred_torsions_1) / norm_scale
        torsions_vf_loss = self._exp_cfg.training.torsions_loss_weight * torch.sum(
            torsions_vf_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom
        return torsions_vf_loss

    def cal_aux_loss_atom(self, gt_bb_atoms, pred_trans_1, pred_rotmats_1, norm_scale, loss_mask):
        # Backbone atom loss
        loss_denom = torch.sum(loss_mask, dim=-1) * 3 + 1e-4
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3]
        gt_bb_atoms_norm = gt_bb_atoms * self._exp_cfg.training.bb_atom_scale / norm_scale[..., None]
        pred_bb_atoms_norm = pred_bb_atoms * self._exp_cfg.training.bb_atom_scale / norm_scale[..., None]
        bb_atom_loss = torch.sum(
            (gt_bb_atoms_norm - pred_bb_atoms_norm) ** 2 * loss_mask[..., None, None],
            dim=(-1, -2, -3)
        ) / loss_denom
        return bb_atom_loss

    def cal_aux_loss_pairdist(self, gt_bb_atoms, pred_trans_1, pred_rotmats_1, norm_scale, loss_mask):
        # Pairwise distance loss
        num_batch, num_res = loss_mask.shape
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3]
        gt_bb_atoms_norm = gt_bb_atoms * self._exp_cfg.training.bb_atom_scale / norm_scale[..., None]
        pred_bb_atoms_norm = pred_bb_atoms * self._exp_cfg.training.bb_atom_scale / norm_scale[..., None]

        gt_flat_atoms = gt_bb_atoms_norm.reshape([num_batch, num_res*3, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_bb_atoms_norm.reshape([num_batch, num_res*3, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res*3])
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res*3])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask,
            dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) + 1)
        return dist_mat_loss

    def cal_sidechain_fape_loss(self, pred_angles, input_batch):
        # build gt
        gt_rigids = du.create_rigid(input_batch['rotmats_1'], input_batch['trans_1'])
        backb_to_global = Rigid(
                                Rotation(
                                    rot_mats=gt_rigids.get_rots().get_rot_mats(), 
                                    quats=None
                                ),
                                gt_rigids.get_trans(),
                            )
        # TODO: if scale is need here
        # backb_to_global = backb_to_global.scale_translation(10)
        bb_anlges = torch.stack((torch.sin(input_batch['bb_torsions_1']), torch.cos(input_batch['bb_torsions_1'])), dim=-1)
        # bb_anlges = torch.zeros(size=(*pred_angles.shape[:2], 3, 2), device=pred_angles.device)
        pred_angles_with_bb_anlges = torch.cat([bb_anlges, pred_angles], dim=2)
        pred_sidechain_frames = torsion_angles_to_frames(r=backb_to_global,
                                                         alpha=pred_angles_with_bb_anlges,
                                                         aatype=input_batch['aatypes_1'],
                                                         rrgdf=self.default_tempalte('default_frames', pred_angles.dtype, pred_angles.device))
        # pred_sidechain_frames = pred_sidechain_frames.to(pred_angles.device)
        pred_atom14_pos = frames_and_literature_positions_to_atom14_pos(pred_sidechain_frames, 
                                                                        input_batch['aatypes_1'],
                                                                        self.default_tempalte('default_frames', pred_angles.dtype, pred_angles.device),
                                                                        self.default_tempalte('group_idx', pred_angles.dtype, pred_angles.device),
                                                                        self.default_tempalte('atom_mask', pred_angles.dtype, pred_angles.device),
                                                                        self.default_tempalte('lit_positions', pred_angles.dtype, pred_angles.device),)
        # pred_atom14_pos = pred_atom14_pos.to(pred_angles.device)
        
        gt_batch = {k:input_batch[k] for k in ('atom14_gt_positions', 'atom14_alt_gt_positions', 'atom14_gt_exists', 'atom14_atom_is_ambiguous', 'atom14_alt_gt_exists')}
        renamed_gt_batch = compute_renamed_ground_truth(batch=gt_batch, atom14_pred_positions=pred_atom14_pos)
        
        sidechain_fape_loss = sidechain_loss(sidechain_frames=pred_sidechain_frames.to_tensor_4x4()[None],
                                            sidechain_atom_pos=pred_atom14_pos[None],
                                            rigidgroups_gt_frames=input_batch["rigidgroups_gt_frames"],
                                            rigidgroups_alt_gt_frames=input_batch["rigidgroups_alt_gt_frames"],
                                            rigidgroups_gt_exists=input_batch["rigidgroups_gt_exists"],
                                            renamed_atom14_gt_positions=renamed_gt_batch["renamed_atom14_gt_positions"],
                                            renamed_atom14_gt_exists=renamed_gt_batch["renamed_atom14_gt_exists"],
                                            alt_naming_is_better=renamed_gt_batch["alt_naming_is_better"],)
        
        return sidechain_fape_loss

    def model_step(self, noisy_batch: Any, training_models=('backbone', )):
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch['res_mask'] * noisy_batch['diffuse_mask'] # [B, L]
        if training_cfg.mask_plddt:
            loss_mask *= noisy_batch['plddt_mask']
        loss_denom = torch.sum(loss_mask, dim=-1) * 3 # [B, ] the number of generated residues for each sample in a batch
        if torch.any(torch.sum(loss_mask, dim=-1) < 1):
            raise ValueError('Empty batch encountered')
        num_batch, num_res = loss_mask.shape

        # Ground truth labels
        gt_trans_1 = noisy_batch['trans_1']
        gt_rotmats_1 = noisy_batch['rotmats_1']
        rotmats_t = noisy_batch['rotmats_t']
        gt_aatypes_1 = noisy_batch['aatypes_1']
        gt_torsions_1 = noisy_batch['torsions_1']

        # Timestep used for normalization
        r3_t = noisy_batch['r3_t'] # (B, 1)
        so3_t = noisy_batch['so3_t'] # (B, 1)
        cat_t = noisy_batch['cat_t'] # (B, 1)
        tor_t = noisy_batch['tor_t'] # (B, 1, 1)

        r3_norm_scale = 1 - torch.min(
            r3_t[..., None], torch.tensor(training_cfg.t_normalize_clip)) # (B, 1, 1)

        so3_norm_scale = 1 - torch.min(
            so3_t[..., None], torch.tensor(training_cfg.t_normalize_clip)) # (B, 1, 1)

        if self._exp_cfg.training.use_torsion_norm_scale:
            tor_norm_scale = 1 - torch.min(
                tor_t[..., None], torch.tensor(training_cfg.t_normalize_clip))
        else:
            tor_norm_scale = 1.0
        
        if training_cfg.aatypes_loss_use_likelihood_weighting:
            cat_norm_scale = 1 - torch.min(
                cat_t, torch.tensor(training_cfg.t_normalize_clip)) # (B, 1)
            assert cat_norm_scale.shape == (num_batch, 1)
        else:
            cat_norm_scale = 1.0
        
        # Model output predictions
        model_output = self.run_model(noisy_batch, until=training_models, mod_t=False)

        # Calculate main loss
        total_loss = {}
        curr_losses = sum([self.model_losses[training_model] for training_model in training_models], [])
        for loss_id in curr_losses:
            if loss_id.startswith('backbone') or loss_id.startswith('refine'): # when backbone model or refine model is trained
                model_type = loss_id.split('_')[0]
                pred_trans_1 = model_output[f'{model_type}_pred_trans']
                pred_rotmats_1 = model_output[f'{model_type}_pred_rotmats']
                pred_logits = model_output[f'{model_type}_pred_logits'] # (B, N, aatype_pred_num_tokens)
                if loss_id.endswith('trans'):
                    trans_loss_mask = loss_mask * r3_t.lt(1)
                    total_loss[loss_id] = self.cal_trans_loss(gt_trans_1, pred_trans_1, r3_norm_scale, trans_loss_mask)
                elif loss_id.endswith('rotmats'):
                    rots_loss_mask = loss_mask * so3_t.lt(1)
                    total_loss[loss_id] = self.cal_rot_loss(gt_rotmats_1, pred_rotmats_1, rotmats_t, so3_norm_scale, rots_loss_mask, so3_t)
                elif loss_id.endswith('aatypes'):
                    aatypes_loss_mask = loss_mask * cat_t.lt(1)
                    total_loss[loss_id] = self.cal_aatype_loss(gt_aatypes_1, pred_logits, cat_norm_scale, aatypes_loss_mask)
            else:
                pred_sin_cos = model_output['sidechain_pred_torsions_sincos']
                pred_sin_cos_unnorm = model_output['sidechain_pred_torsions_sincos_unnorm']
                gt_sin_cos = torch.stack((torch.sin(gt_torsions_1), torch.cos(gt_torsions_1)), dim=-1)
                total_loss[loss_id] = supervised_chi_loss(angles_sin_cos=pred_sin_cos,
                                                        unnormalized_angles_sin_cos=pred_sin_cos_unnorm,
                                                        aatype=gt_aatypes_1,
                                                        seq_mask=loss_mask,
                                                        chi_mask=noisy_batch['torsions_mask'],
                                                        chi_angles_sin_cos=gt_sin_cos,
                                                        chi_weight=1,
                                                        angle_norm_weight=0.02,
                                                        eps=1e-6,)
                # for packing model in stage 1
                if self._exp_cfg.train_packing_only:
                    sidechain_fape_loss = self.cal_sidechain_fape_loss(pred_sin_cos, noisy_batch)
                    total_loss['sidechain_fape'] = sidechain_fape_loss

        train_loss = sum(total_loss[loss_id] for loss_id in total_loss)

        # Calculate auxiliary loss and FAPE loss
        if 'refine' in training_models:
            curr_aux_loss_weight = training_cfg.refine_aux_loss_weight
            curr_fape_loss_weight = training_cfg.refine_fape_loss_weight
            pred_rigids_1 = model_output['refine_pred_rigids']

        else:
            curr_aux_loss_weight = training_cfg.backbone_aux_loss_weight
            curr_fape_loss_weight = 0
        
        if len(training_models) == 1 and training_models[0] == 'sidechain': # means only sidechain model is trained
            curr_aux_loss_weight = 0
            curr_fape_loss_weight = 0
        
        if curr_aux_loss_weight > 0:
            auxiliary_loss = None
            gt_bb_atoms = all_atom.to_atom37(gt_trans_1, gt_rotmats_1)[:, :, :3]
            if training_cfg.aux_loss_use_bb_loss:
                bb_atom_loss = self.cal_aux_loss_atom(gt_bb_atoms, pred_trans_1, pred_rotmats_1, r3_norm_scale, trans_loss_mask)
                auxiliary_loss = bb_atom_loss
            if training_cfg.aux_loss_use_pair_loss:
                pair_dist_loss = self.cal_aux_loss_pairdist(gt_bb_atoms, pred_trans_1, pred_rotmats_1, r3_norm_scale, trans_loss_mask)
                if auxiliary_loss is None:
                    auxiliary_loss = pair_dist_loss
                else:
                    auxiliary_loss += pair_dist_loss
        
            if not auxiliary_loss is None:
                auxiliary_loss *= (
                    (r3_t[:, 0] > training_cfg.aux_loss_t_pass)
                    & (so3_t[:, 0] > training_cfg.aux_loss_t_pass)
                )
                auxiliary_loss *= curr_aux_loss_weight
                auxiliary_loss = torch.clamp(auxiliary_loss, max=5)
                train_loss += auxiliary_loss
                total_loss[f'{training_models[-1]}_auxiliary'] = auxiliary_loss
        
        if curr_fape_loss_weight > 0:
            gt_backbone_rigids = du.create_rigid(gt_rotmats_1, gt_trans_1)
            fape_loss = backbone_loss(backbone_rigid_tensor=gt_backbone_rigids.to_tensor_4x4(),
                                      backbone_rigid_mask=trans_loss_mask,
                                      traj=pred_rigids_1.to_tensor_7(),
                                      use_clamped_fape=None,
                                      clamp_distance=10.0,
                                      loss_unit_distance=10.0,
                                      eps=1e-4,)
            fape_loss = fape_loss * curr_fape_loss_weight
            train_loss += fape_loss
            total_loss['refine_fape'] = fape_loss

        # train_loss = train_loss * min(1, np.sqrt(num_res/384)) # reweight the loss according to the number of residues
        total_loss['train_loss'] = train_loss
        
        if torch.any(torch.isnan(train_loss)):
            raise ValueError('NaN loss encountered')
        self._prev_batch = noisy_batch
        self._prev_loss_denom = loss_denom
        self._prev_loss = total_loss
        # print('#'*3, self.global_step, '#'*3 , training_models)#, total_loss.keys())
        # for term in noisy_batch:
        #     if term in ('cat_t', 'so3_t', 'r3_t', 'tor_t'):
        #         print(term, noisy_batch[term])

        return self._prev_loss, model_output

    def backbone_model_step(self, noisy_batch: Any):
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch['res_mask'] * noisy_batch['diffuse_mask'] # [B, L]
        # if training_cfg.mask_plddt:
        #     loss_mask *= noisy_batch['plddt_mask']
        loss_denom = torch.sum(loss_mask, dim=-1) * 3 # [B, ] the number of generated residues for each sample in a batch
        if torch.any(torch.sum(loss_mask, dim=-1) < 1):
            raise ValueError('Empty batch encountered')
        num_batch, num_res = loss_mask.shape

        # Ground truth labels
        gt_trans_1 = noisy_batch['trans_1']
        gt_rotmats_1 = noisy_batch['rotmats_1']
        rotmats_t = noisy_batch['rotmats_t']
        gt_aatypes_1 = noisy_batch['aatypes_1']
        gt_torsions_1 = noisy_batch['torsions_1']

        # Timestep used for normalization
        r3_t = noisy_batch['r3_t'] # (B, 1)
        so3_t = noisy_batch['so3_t'] # (B, 1)
        cat_t = noisy_batch['cat_t'] # (B, 1)
        tor_t = noisy_batch['tor_t'] # (B, 1, 1)

        r3_norm_scale = 1 - torch.min(
            r3_t[..., None], torch.tensor(training_cfg.t_normalize_clip)) # (B, 1, 1)

        so3_norm_scale = 1 - torch.min(
            so3_t[..., None], torch.tensor(training_cfg.t_normalize_clip)) # (B, 1, 1)

        if self._exp_cfg.training.use_torsion_norm_scale:
            tor_norm_scale = 1 - torch.min(
                tor_t[..., None], torch.tensor(training_cfg.t_normalize_clip))
        else:
            tor_norm_scale = 1.0
        
        if training_cfg.aatypes_loss_use_likelihood_weighting:
            cat_norm_scale = 1 - torch.min(
                cat_t, torch.tensor(training_cfg.t_normalize_clip)) # (B, 1)
            assert cat_norm_scale.shape == (num_batch, 1)
        else:
            cat_norm_scale = 1.0
        
        # Model output predictions
        model_output = self.model['backbone'](noisy_batch)

        # Calculate main loss
        total_loss = {}
        aatypes_loss_mask = loss_mask * cat_t.lt(1)
        total_loss['backbone_aatypes'] = self.cal_aatype_loss(gt_aatypes_1, model_output[f'backbone_pred_logits'] , cat_norm_scale, aatypes_loss_mask)
        trans_loss_mask = loss_mask * r3_t.lt(1)
        total_loss['backbone_trans'] = self.cal_trans_loss(gt_trans_1, model_output[f'backbone_pred_trans'], r3_norm_scale, trans_loss_mask)
        rots_loss_mask = loss_mask * so3_t.lt(1)
        total_loss['backbone_rotmats'] = self.cal_rot_loss(gt_rotmats_1, model_output[f'backbone_pred_rotmats'], rotmats_t, so3_norm_scale, rots_loss_mask, so3_t)
        train_loss = total_loss['backbone_aatypes'] + total_loss['backbone_trans'] + total_loss['backbone_rotmats']

        # Calculate auxiliary loss
        curr_aux_loss_weight = training_cfg.backbone_aux_loss_weight

        if curr_aux_loss_weight > 0:
            auxiliary_loss = None
            gt_bb_atoms = all_atom.to_atom37(gt_trans_1, gt_rotmats_1)[:, :, :3]
            if training_cfg.aux_loss_use_bb_loss:
                bb_atom_loss = self.cal_aux_loss_atom(gt_bb_atoms, pred_trans_1, pred_rotmats_1, r3_norm_scale, trans_loss_mask)
                auxiliary_loss = bb_atom_loss
            if training_cfg.aux_loss_use_pair_loss:
                pair_dist_loss = self.cal_aux_loss_pairdist(gt_bb_atoms, pred_trans_1, pred_rotmats_1, r3_norm_scale, trans_loss_mask)
                if auxiliary_loss is None:
                    auxiliary_loss = pair_dist_loss
                else:
                    auxiliary_loss += pair_dist_loss
        
            if not auxiliary_loss is None:
                auxiliary_loss *= (
                    (r3_t[:, 0] > training_cfg.aux_loss_t_pass)
                    & (so3_t[:, 0] > training_cfg.aux_loss_t_pass)
                )
                auxiliary_loss *= curr_aux_loss_weight
                auxiliary_loss = torch.clamp(auxiliary_loss, max=5)
                train_loss += auxiliary_loss
                total_loss['auxiliary'] = auxiliary_loss

        # train_loss = train_loss * min(1, np.sqrt(num_res/384)) # reweight the loss according to the number of residues
        total_loss['train_loss'] = train_loss
        
        if torch.any(torch.isnan(train_loss)):
            raise ValueError('NaN loss encountered')
        self._prev_batch = noisy_batch
        self._prev_loss_denom = loss_denom
        self._prev_loss = total_loss
        return self._prev_loss, model_output

    def validation_step(self, batch: Any, batch_idx: int):
        res_mask = batch['res_mask']
        self.interpolant.set_device(res_mask.device)
        num_batch, num_res = res_mask.shape
        assert num_batch == 1
        
        diffuse_mask = batch['diffuse_mask']
        csv_idx = batch['csv_idx']

        assert (diffuse_mask == 1.0).all()

        prot_traj, model_traj, torsion_traj, clean_torsion_traj, gt_result = self.interpolant.sample(
            num_batch,
            num_res,
            self.model,
            self.model_order,
            trans_1=batch['trans_1'],
            rotmats_1=batch['rotmats_1'],
            aatypes_1=batch['aatypes_1'],
            torsions_1=batch['torsions_1'],
            diffuse_mask=diffuse_mask,
            chain_idx=batch['chain_idx'],
            res_idx=batch['res_idx'],
            forward_folding=True if self._exp_cfg.validation.task == 'forward_folding' else False,
            inverse_folding=True if self._exp_cfg.validation.task == 'inverse_folding' else False,
            packing=True if self._exp_cfg.validation.task == 'packing' else False,
            separate_t=True if self._exp_cfg.validation.task != 'codesign' else False,
            esm_model=self.folding_model,
        )
        samples = prot_traj[-1][0].numpy()
        samples_full = prot_traj[-1][2].numpy()
        assert samples.shape == (num_batch, num_res, 37, 3)
        # assert False, "need to separate aatypes from atom37_traj"

        generated_aatypes = prot_traj[-1][1].numpy()
        assert generated_aatypes.shape == (num_batch, num_res)
        batch_level_aatype_metrics = mu.calc_aatype_metrics(generated_aatypes)

        batch_metrics = []
        for i in range(num_batch):
            sample_dir = os.path.join(
                self.checkpoint_dir,
                f'epoch_{self.current_epoch}',
                f'sample_{csv_idx[i].item()}_idx_{batch_idx}_len_{num_res}_process_{torch.cuda.current_device()}'
            )
            os.makedirs(sample_dir, exist_ok=True)

            # Write out sample to PDB file
            final_pos = samples[i]
            saved_path = au.write_prot_to_pdb(
                final_pos,
                os.path.join(sample_dir, 'sample.pdb'),
                aatype=generated_aatypes[i],
                no_indexing=True
            )

            # Write out [Full-Atom] sample to PDB file
            final_pos_full = samples_full[i]
            saved_path_full = au.write_prot_to_pdb(
                final_pos_full,
                os.path.join(sample_dir, 'sample_full.pdb'),
                aatype=generated_aatypes[i],
                no_indexing=True
            )
            
            if self._interpolant_cfg.save_gt:
                gt_pos = gt_result['gt_atom37'][i].cpu().numpy()
                gt_aatypes = gt_result['gt_aatypes'][i].cpu().numpy()
                gt_pos_full = gt_result['gt_atom37_full'][i].cpu().numpy()
                
                # Write out gt to PDB file
                saved_path = au.write_prot_to_pdb(
                    gt_pos,
                    os.path.join(sample_dir, f'gt_{batch["pdb_name"][i]}_backbone.pdb'),
                    aatype=gt_aatypes,
                    no_indexing=True
                )
                
                # Write out [Full-Atom] gt to PDB file
                saved_path_full = au.write_prot_to_pdb(
                    gt_pos_full,
                    os.path.join(sample_dir, f'gt_{batch["pdb_name"][i]}_full.pdb'),
                    aatype=gt_aatypes,
                    no_indexing=True
                )         
            
            fullatom_trajs = du.to_numpy(torch.cat([x[2][i:i+1] for x in prot_traj[::-1]], dim=0))
            aatypes_trajs = du.to_numpy(torch.cat([x[1][i:i+1] for x in prot_traj[::-1]], dim=0).long())
            clean_aatypes_trajs = copy.deepcopy(aatypes_trajs)
            clean_aatypes_trajs[clean_aatypes_trajs == du.MASK_TOKEN_INDEX] = 0
            model_fullatom_trajs = du.to_numpy(torch.cat([x[2][i:i+1] for x in model_traj[::-1]], dim=0))
            model_aatypes_trajs = du.to_numpy(torch.cat([x[1][i:i+1] for x in model_traj[::-1]], dim=0).long())
            clean_model_aatypes_trajs = copy.deepcopy(model_aatypes_trajs)
            clean_model_aatypes_trajs[clean_model_aatypes_trajs == du.MASK_TOKEN_INDEX] = 0
            # Write out [Full-Atom] sample [trajectories] to PDB file
            _ = au.write_prot_to_pdb(
                fullatom_trajs,
                os.path.join(sample_dir, 'sample_full_traj.pdb'),
                aatype=clean_aatypes_trajs,
                no_indexing=True
            )
            # Write out [Full-Atom] model prediction [trajectories] to PDB file
            _ = au.write_prot_to_pdb(
                model_fullatom_trajs,
                os.path.join(sample_dir, 'model_full_traj.pdb'),
                aatype=clean_model_aatypes_trajs,
                no_indexing=True
            )

            if isinstance(self.logger, WandbLogger):
                self.validation_epoch_samples.append(
                    [saved_path, self.global_step, wandb.Molecule(saved_path)]
                )

            try:
                # Run designability
                pmpnn_pdb_path = saved_path.replace('.pdb', '_pmpnn.pdb')
                shutil.copy(saved_path, pmpnn_pdb_path)
                pmpnn_fasta_path = self.run_pmpnn(
                    sample_dir,
                    pmpnn_pdb_path,
                )
                folded_dir = os.path.join(sample_dir, 'folded')
                os.makedirs(folded_dir, exist_ok=True)

                if self.interpolant._aatypes_cfg.corrupt:
                    # Codesign
                    codesign_fasta = fasta.FastaFile()
                    codesign_fasta['codesign_seq_1'] = "".join([restypes_with_x[x] for x in generated_aatypes[i]])
                    codesign_fasta_path = os.path.join(sample_dir, 'codesign.fa')
                    codesign_fasta.write(codesign_fasta_path)

                    codesign_folded_output = self.folding_model.fold_fasta(codesign_fasta_path, folded_dir)
                    codesign_results = mu.process_folded_outputs(saved_path, codesign_folded_output)

                    # make a fasta file with a single PMPNN sequence to be folded
                    reloaded_fasta = fasta.FastaFile.read(pmpnn_fasta_path)
                    single_fasta = fasta.FastaFile()
                    single_fasta['pmpnn_seq_1'] = reloaded_fasta['pmpnn_seq_1']
                    single_fasta_path = os.path.join(sample_dir, 'pmpnn_single.fasta')
                    single_fasta.write(single_fasta_path)

                    single_pmpnn_folded_output = self.folding_model.fold_fasta(single_fasta_path, folded_dir)
                    single_pmpnn_results = mu.process_folded_outputs(saved_path, single_pmpnn_folded_output)

                    designable_metrics = {
                        'codesign_bb_rmsd': codesign_results.bb_rmsd.min(),
                        'pmpnn_bb_rmsd': single_pmpnn_results.bb_rmsd.min(),
                    }
                else:
                    # Just structure
                    folded_output = self.folding_model.fold_fasta(pmpnn_fasta_path, folded_dir)

                    designable_results = mu.process_folded_outputs(saved_path, folded_output) 
                    designable_metrics = {
                        'bb_rmsd': designable_results.bb_rmsd.min()
                    }
                try:
                    mdtraj_metrics = mu.calc_mdtraj_metrics(saved_path)
                    ca_ca_metrics = mu.calc_ca_ca_metrics(final_pos[:, residue_constants.atom_order['CA']])
                    batch_metrics.append((mdtraj_metrics | ca_ca_metrics | designable_metrics | batch_level_aatype_metrics))
                except Exception as e:
                    print(e)
                    continue
            except:
                continue
        
        if len(batch_metrics) > 0:
            batch_metrics = pd.DataFrame(batch_metrics)
            self.validation_epoch_metrics.append(batch_metrics)
        
    def on_validation_epoch_end(self):
        if len(self.validation_epoch_samples) > 0:
            self.logger.log_table(
                key='valid/samples',
                columns=["sample_path", "current_epoch", "global_step", "Protein", "Protein_full"],
                data=self.validation_epoch_samples)
            self.validation_epoch_samples.clear()
        if len(self.validation_epoch_metrics) > 0:
            val_epoch_metrics = pd.concat(self.validation_epoch_metrics)
            for metric_name,metric_val in val_epoch_metrics.mean().to_dict().items():
                self._log_scalar(
                    f'valid/{metric_name}',
                    metric_val,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=len(val_epoch_metrics),
                )
        self.validation_epoch_metrics.clear()
        # self._folding_model.to_cpu()
        # self.folding_model.to_cpu()

    def _log_scalar(
            self,
            key,
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=None,
            sync_dist=False,
            rank_zero_only=True
        ):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )

    def update_self_condition(self, output_noisy_batch, ):
        # for the three backbone data modalities, aatypes, trans, and rots, there are two sources: 1. backbone model 2. refine model
        # so the self-condition consist of three parts: 1. backbone model 2. refine model 3. previous self-condition
        #     if refine model is available, and t has reached the refine model start time, we use refine model output as the self-condition
        #     elif t has not reached the backbone model start time, we use backbone model output as the self-condition
        #     else, we use previous self-condition
        # another situtation is that when executing specific tasks, like folding or inverse folding, the t of some modality is 1
        #     in this case, we should also use the previous self-condition
        
        if 'refine_model_mask' not in output_noisy_batch:
            if 'refine_pred_trans' in output_noisy_batch:
                refine_model_mask = torch.ones_like(output_noisy_batch['r3_t']) # [B, 1]
            else: # which means refine model does not execute in calculating the current output_noisy_batch
                refine_model_mask = torch.zeros_like(output_noisy_batch['r3_t']) # [B, 1]
                output_noisy_batch['refine_pred_trans'] = copy.deepcopy(output_noisy_batch['backbone_pred_trans']) # placeholder
                output_noisy_batch['refine_pred_rotmats'] = copy.deepcopy(output_noisy_batch['backbone_pred_rotmats']) # placeholder
                output_noisy_batch['refine_pred_logits'] = copy.deepcopy(output_noisy_batch['backbone_pred_logits']) # placeholder
        else:
            refine_model_mask = output_noisy_batch['refine_model_mask']
        
        # update trans self-condition
        trans_1_mask = output_noisy_batch['r3_t'].ge(1).float()
        trans_refine_mask = output_noisy_batch['r3_t'].gt(self.model_start_t['refine']).float() - \
                            trans_1_mask # [B, 1], for the sample that has reached the refine model start time and not reached 1, it's trans_refine_mask is 1
        trans_refine_mask = trans_refine_mask * refine_model_mask # [B, 1], if refine model not exists, trans_refine_mask is all 0
        trans_prevsc_mask = output_noisy_batch['r3_t'].lt(self.model_start_t['backbone']).float() + \
                            trans_1_mask # [B, 1], for the sample that has not reached the backbone model start time or reach 1, it's trans_prevsc_mask is 1
        trans_backbone_mask = 1 - trans_refine_mask - trans_prevsc_mask # otherwise, will use backbone model output
        trans_sc = output_noisy_batch['refine_pred_trans'] * trans_refine_mask[..., None] + \
                   output_noisy_batch['backbone_pred_trans'] * trans_backbone_mask[..., None] + \
                   output_noisy_batch['trans_sc'] * trans_prevsc_mask[..., None]
        output_noisy_batch['trans_sc'] = (
            trans_sc * output_noisy_batch['diffuse_mask'][..., None]
            + output_noisy_batch['trans_1'] * (1 - output_noisy_batch['diffuse_mask'][..., None])
        )

        # update rotvecs self-condition
        rots_1_mask = output_noisy_batch['so3_t'].ge(1).float()
        rots_refine_mask = output_noisy_batch['so3_t'].gt(self.model_start_t['refine']).float() - \
                           rots_1_mask # [B, 1]
        rots_refine_mask = rots_refine_mask * refine_model_mask
        rots_prevsc_mask = output_noisy_batch['so3_t'].lt(self.model_start_t['backbone']).float() + \
                           rots_1_mask # [B, 1]
        rots_backbone_mask = 1 - trans_refine_mask - rots_prevsc_mask
        rotvecs_sc = so3_utils.rotmat_to_rotvec(output_noisy_batch['refine_pred_rotmats']) * rots_refine_mask[..., None] + \
                     so3_utils.rotmat_to_rotvec(output_noisy_batch['backbone_pred_rotmats']) * rots_backbone_mask[..., None] + \
                     output_noisy_batch['rotvecs_sc'] * rots_prevsc_mask[..., None]
        output_noisy_batch['rotvecs_sc'] = (
            rotvecs_sc * output_noisy_batch['diffuse_mask'][..., None]
            + so3_utils.rotmat_to_rotvec(output_noisy_batch['rotmats_1']) * (1 - output_noisy_batch['diffuse_mask'][..., None])
        )

        # update aatypes self-condition
        aatypes_1_mask = output_noisy_batch['cat_t'].ge(1).float()
        aatypes_refine_mask = output_noisy_batch['cat_t'].gt(self.model_start_t['refine']).float() - \
                              aatypes_1_mask # [B, 1]
        aatypes_refine_mask = aatypes_refine_mask * refine_model_mask
        aatypes_prevsc_mask = output_noisy_batch['cat_t'].lt(self.model_start_t['backbone']).float() + \
                              aatypes_1_mask # [B, 1]
        aatypes_backbone_mask = 1 - aatypes_refine_mask - aatypes_prevsc_mask
        aatypes_sc = output_noisy_batch['refine_pred_logits'] * aatypes_refine_mask[..., None] + \
                     output_noisy_batch['backbone_pred_logits'] * aatypes_backbone_mask[..., None] + \
                     output_noisy_batch['aatypes_sc'] * aatypes_prevsc_mask[...,  None]
        output_noisy_batch['aatypes_sc'] = (
            aatypes_sc * output_noisy_batch['diffuse_mask'][..., None]
            + output_noisy_batch['logits_1'] * (1 - output_noisy_batch['diffuse_mask'][..., None])
        )

        # update torsions self-condition
        if 'sidechain_pred_torsions' in output_noisy_batch:
            pred_torsions_1 = so2_utils.mod_to_standard_angle_range(output_noisy_batch['sidechain_pred_torsions'])
            torsions_update_mask = output_noisy_batch['tor_t'].gt(self.model_start_t['sidechain']).float() - \
                                   output_noisy_batch['tor_t'].ge(1).float() # in any situation, tor_t will not exceeds 1, we just maintain consistency in the code style here
            torsions_update_mask = torsions_update_mask[..., None] # [B, 1, 1]
            torsions_sc = pred_torsions_1 * torsions_update_mask + output_noisy_batch['torsions_sc'] * (1 - torsions_update_mask) # keep the previous self-condition if t has not reached the model start time
            output_noisy_batch['torsions_sc'] = (
                torsions_sc * output_noisy_batch['diffuse_mask'][..., None] 
                + output_noisy_batch['torsions_1']* (1 - output_noisy_batch['diffuse_mask'][..., None])
            )
        
        output_noisy_batch['refine_model_mask'] = refine_model_mask
        return output_noisy_batch
    
    def update_self_condition_woRefine(self, output_noisy_batch, ):
        # used for updating self-condition when there is no refine model
    
        # update t
        trans_1_mask = output_noisy_batch['r3_t'].ge(1).float()
        trans_prevsc_mask = output_noisy_batch['r3_t'].lt(self.model_start_t['backbone']).float() + \
                            trans_1_mask # [B, 1], for the sample that has not reached the backbone model start time or reach 1, it's trans_prevsc_mask is 1
        trans_backbone_mask = 1 - trans_prevsc_mask # otherwise, will use backbone model output
        trans_sc = output_noisy_batch['backbone_pred_trans'] * trans_backbone_mask[..., None] + \
                   output_noisy_batch['trans_sc'] * trans_prevsc_mask[..., None]
        output_noisy_batch['trans_sc'] = (
            trans_sc * output_noisy_batch['diffuse_mask'][..., None]
            + output_noisy_batch['trans_1'] * (1 - output_noisy_batch['diffuse_mask'][..., None])
        )

        # update rotvecs self-condition
        rots_1_mask = output_noisy_batch['so3_t'].ge(1).float()
        rots_prevsc_mask = output_noisy_batch['so3_t'].lt(self.model_start_t['backbone']).float() + \
                           rots_1_mask # [B, 1]
        rots_backbone_mask = 1 - rots_prevsc_mask
        rotvecs_sc = so3_utils.rotmat_to_rotvec(output_noisy_batch['backbone_pred_rotmats']) * rots_backbone_mask[..., None] + \
                     output_noisy_batch['rotvecs_sc'] * rots_prevsc_mask[..., None]
        output_noisy_batch['rotvecs_sc'] = (
            rotvecs_sc * output_noisy_batch['diffuse_mask'][..., None]
            + so3_utils.rotmat_to_rotvec(output_noisy_batch['rotmats_1']) * (1 - output_noisy_batch['diffuse_mask'][..., None])
        )

        # update aatypes self-condition
        aatypes_1_mask = output_noisy_batch['cat_t'].ge(1).float()
        aatypes_prevsc_mask = output_noisy_batch['cat_t'].lt(self.model_start_t['backbone']).float() + \
                              aatypes_1_mask # [B, 1]
        aatypes_backbone_mask = 1 - aatypes_prevsc_mask
        aatypes_sc = output_noisy_batch['backbone_pred_logits'] * aatypes_backbone_mask[..., None] + \
                     output_noisy_batch['aatypes_sc'] * aatypes_prevsc_mask[...,  None]
        output_noisy_batch['aatypes_sc'] = (
            aatypes_sc * output_noisy_batch['diffuse_mask'][..., None]
            + output_noisy_batch['logits_1'] * (1 - output_noisy_batch['diffuse_mask'][..., None])
        )

        # update torsions self-condition
        if 'sidechain_pred_torsions' in output_noisy_batch:
            pred_torsions_1 = so2_utils.mod_to_standard_angle_range(output_noisy_batch['sidechain_pred_torsions'])
            torsions_update_mask = output_noisy_batch['tor_t'].gt(self.model_start_t['sidechain']).float() - \
                                   output_noisy_batch['tor_t'].ge(1).float() # in any situation, tor_t will not exceeds 1, we just maintain consistency in the code style here
            torsions_update_mask = torsions_update_mask[..., None] # [B, 1, 1]
            torsions_sc = pred_torsions_1 * torsions_update_mask + output_noisy_batch['torsions_sc'] * (1 - torsions_update_mask) # keep the previous self-condition if t has not reached the model start time
            output_noisy_batch['torsions_sc'] = (
                torsions_sc * output_noisy_batch['diffuse_mask'][..., None] 
                + output_noisy_batch['torsions_1']* (1 - output_noisy_batch['diffuse_mask'][..., None])
            )
        
        return output_noisy_batch

    def update_noised_data(self, output_noisy_batch, d_t):
        # similar to the strategy in the update_selfcondition()
        # nosied_data will be updated if the corresponding t has reached the model start time
        # data with t=1 will be kept as the previous state
        if 'refine_model_mask' not in output_noisy_batch:
            if 'refine_pred_trans' in output_noisy_batch:
                refine_model_mask = torch.ones_like(output_noisy_batch['r3_t']) # [B, 1]
            else: # which means refine model does not execute in calculating the current output_noisy_batch
                refine_model_mask = torch.zeros_like(output_noisy_batch['r3_t']) # [B, 1]
                output_noisy_batch['refine_pred_trans'] = copy.deepcopy(output_noisy_batch['backbone_pred_trans']) # placeholder
                output_noisy_batch['refine_pred_rotmats'] = copy.deepcopy(output_noisy_batch['backbone_pred_rotmats']) # placeholder
                output_noisy_batch['refine_pred_logits'] = copy.deepcopy(output_noisy_batch['backbone_pred_logits']) # placeholder
        else:
            refine_model_mask = output_noisy_batch['refine_model_mask']

        # update trans_t
        refine_trans_t_2 = self.interpolant._trans_euler_step(d_t, output_noisy_batch['r3_t'][..., None], output_noisy_batch['refine_pred_trans'], copy.deepcopy(output_noisy_batch['trans_t']))
        backbone_trans_t_2 = self.interpolant._trans_euler_step(d_t, output_noisy_batch['r3_t'][..., None], output_noisy_batch['backbone_pred_trans'], copy.deepcopy(output_noisy_batch['trans_t']))
        trans_refine_mask = output_noisy_batch['r3_t'].gt(self.model_start_t['refine']).float() - \
                            output_noisy_batch['r3_t'].ge(1).float()
        trans_refine_mask = trans_refine_mask * refine_model_mask # [B, 1]
        trans_prevt_mask = output_noisy_batch['r3_t'].lt(self.model_start_t['backbone']).float() + \
                           output_noisy_batch['r3_t'].ge(1).float() # [B, 1]
        trans_backbone_mask = 1 - trans_refine_mask - trans_prevt_mask
        trans_t_2 = refine_trans_t_2 * trans_refine_mask[..., None] + \
                    backbone_trans_t_2 * trans_backbone_mask[..., None] + \
                    output_noisy_batch['trans_t'] * trans_prevt_mask[..., None]
        output_noisy_batch['trans_t'] = (
            trans_t_2 * output_noisy_batch['diffuse_mask'][..., None]
            + output_noisy_batch['trans_1'] * (1 - output_noisy_batch['diffuse_mask'][..., None])
        )
        r3_t = output_noisy_batch['r3_t'] + d_t
        output_noisy_batch['r3_t'] = r3_t.clamp(-1, 1)

        # update rotmats_t
        refine_rotmats_t_2 = self.interpolant._rots_euler_step(d_t, output_noisy_batch['so3_t'][..., None], output_noisy_batch['refine_pred_rotmats'], output_noisy_batch['rotmats_t'])
        backbone_rotmats_t_2 = self.interpolant._rots_euler_step(d_t, output_noisy_batch['so3_t'][..., None], output_noisy_batch['backbone_pred_rotmats'], output_noisy_batch['rotmats_t'])
        rots_refine_mask = output_noisy_batch['so3_t'].gt(self.model_start_t['refine']).float() - \
                           output_noisy_batch['so3_t'].ge(1).float()
        rots_refine_mask = rots_refine_mask * refine_model_mask
        rots_prevt_mask = output_noisy_batch['so3_t'].lt(self.model_start_t['backbone']).float() + \
                          output_noisy_batch['so3_t'].ge(1).float()
        rots_backbone_mask = 1 - rots_refine_mask - rots_prevt_mask
        rotmats_t_2 = refine_rotmats_t_2 * rots_refine_mask[..., None, None] + \
                      backbone_rotmats_t_2 * rots_backbone_mask[..., None, None] + \
                      output_noisy_batch['rotmats_t'] * rots_prevt_mask[..., None, None]
        output_noisy_batch['rotmats_t'] = (
            rotmats_t_2 * output_noisy_batch['diffuse_mask'][..., None, None]
            + output_noisy_batch['rotmats_1'] * (1 - output_noisy_batch['diffuse_mask'][..., None, None])
        )
        so3_t = output_noisy_batch['so3_t'] + d_t
        output_noisy_batch['so3_t'] = so3_t.clamp(-1, 1)

        # update aatypes_t
        refine_aatypes_t_2 = self.interpolant._aatypes_euler_step_purity_refine(d_t, output_noisy_batch['cat_t'].squeeze(), output_noisy_batch['backbone_pred_logits'], output_noisy_batch['refine_pred_logits'], output_noisy_batch['aatypes_t'])
        backbone_aatypes_t_2 = self.interpolant._aatypes_euler_step_purity(d_t, output_noisy_batch['cat_t'].squeeze(), output_noisy_batch['backbone_pred_logits'], output_noisy_batch['aatypes_t'])
        aatypes_refine_mask = output_noisy_batch['cat_t'].gt(self.model_start_t['refine']).float() - \
                              output_noisy_batch['cat_t'].ge(1).float()
        aatypes_refine_mask = aatypes_refine_mask * refine_model_mask
        aatypes_prevt_mask = output_noisy_batch['cat_t'].lt(self.model_start_t['backbone']).float() + \
                             output_noisy_batch['cat_t'].ge(1).float() # [B, 1]
        aatypes_backbone_mask = 1 - aatypes_refine_mask - aatypes_prevt_mask
        aatypes_t_2 = refine_aatypes_t_2 * aatypes_refine_mask + \
                      backbone_aatypes_t_2 * aatypes_backbone_mask + \
                      output_noisy_batch['aatypes_t'] * aatypes_prevt_mask
        output_noisy_batch['aatypes_t'] = (
            aatypes_t_2 * output_noisy_batch['diffuse_mask']
            + output_noisy_batch['aatypes_1'] * (1 - output_noisy_batch['diffuse_mask'])
        )
        cat_t = output_noisy_batch['cat_t'] + d_t
        output_noisy_batch['cat_t'] = cat_t.clamp(-1, 1)

        # update torsions_t
        if 'sidechain_pred_torsions' in output_noisy_batch:
            pred_torsions_1 = so2_utils.mod_to_standard_angle_range(output_noisy_batch['sidechain_pred_torsions'])
            sidechain_torsions_t_2 = self.interpolant._torsions_euler_step(d_t, output_noisy_batch['tor_t'][..., None], pred_torsions_1, output_noisy_batch['torsions_t'])
            torsions_refine_mask = output_noisy_batch['tor_t'].gt(self.model_start_t['sidechain']).float() - \
                                   output_noisy_batch['tor_t'].ge(1).float() # in any situation, tor_t will not exceeds 1, we just maintain consistency in the code style here
            torsions_refine_mask = torsions_refine_mask[..., None]
            torsions_t_2 = sidechain_torsions_t_2 * torsions_refine_mask + output_noisy_batch['torsions_t'] * (1 - torsions_refine_mask)
            output_noisy_batch['torsions_t'] = (
                torsions_t_2 * output_noisy_batch['diffuse_mask'][..., None]
                + output_noisy_batch['torsions_1'] * (1 - output_noisy_batch['diffuse_mask'][..., None])
            )
            tor_t = output_noisy_batch['tor_t'] + d_t
            output_noisy_batch['tor_t'] = tor_t.clamp(-1, 1)
        
        output_noisy_batch['refine_model_mask'] = refine_model_mask
        return output_noisy_batch

    def curr_training_model(self, ):
        curr_training_step_ = (self.global_step+1) % self.training_loop_steps
        if curr_training_step_ == 0 or len(self.model_training_orders) == 1:
            curr_training_model = self.model_training_orders[-1]
        else:
            stack_steps = 0
            for model_block in self.model_training_orders:
                stack_steps += self.model_training_steps[model_block]
                if curr_training_step_ <= stack_steps:
                    curr_training_model = model_block
                    break
        return curr_training_model.split(',')
    
    def run_backbone_model_only(self, noisy_batch):
        # calculate and update self-condition
        if self._interpolant_cfg.self_condition >= random.random():
            with torch.no_grad():
                sc_output = self.model['backbone'](noisy_batch)

                noisy_batch['trans_sc'] = (
                    sc_output['backbone_pred_trans'] * noisy_batch['diffuse_mask'][..., None]
                    + noisy_batch['trans_1'] * (1 - noisy_batch['diffuse_mask'][..., None])
                )

                noisy_batch['rotvecs_sc'] = (
                    so3_utils.rotmat_to_rotvec(sc_output['backbone_pred_rotmats']) * noisy_batch['diffuse_mask'][..., None]
                    + so3_utils.rotmat_to_rotvec(noisy_batch['rotmats_1']) * (1 - noisy_batch['diffuse_mask'][..., None])
                )

                noisy_batch['aatypes_sc'] = (
                    sc_output['backbone_pred_logits'] * noisy_batch['diffuse_mask'][..., None]
                    + noisy_batch['logits_1'] * (1 - noisy_batch['diffuse_mask'][..., None])
                )

        # forward model
        batch_losses, model_output = self.backbone_model_step(noisy_batch)
        return batch_losses, model_output
        
    def training_step(self, batch: Any, stage: int):
        
        curr_step_training_models = self.curr_training_model()
        if type(curr_step_training_models) == str:
            curr_step_training_models = (curr_step_training_models, )

        if self.consistency_mode and curr_step_training_models[0] == 'backbone':
            curr_consistency_mode = True
        else:
            curr_consistency_mode = False

        # Set up the gradient for the current training step
        for model_type in self.model_order:
            if model_type in curr_step_training_models:
                self.model[model_type].requires_grad_(True)
            else:
                self.model[model_type].requires_grad_(False)
        
        # curr_step_training_models = ['backbone', ]

        if self.current_epoch > self._exp_cfg.rollout_start_epoch and curr_step_training_models==['backbone', ]: # as only the backbone is Flow model, rollout will only apply to backbone model training
            curr_step_rollout_num = self._exp_cfg.rollout_num
        else:
            curr_step_rollout_num = 0

        step_start_time = time.time()
        self.interpolant.set_device(batch['res_mask'].device)
        noisy_batch = self.interpolant.corrupt_batch(batch, rot_style=self._exp_cfg.rot_training_style, training_models=curr_step_training_models, forward_dt=self.d_t, forward_steps=curr_step_rollout_num)

        # execute PLM
        if not self.use_PLM is None: 
            if self.use_PLM.startswith('gLM'):
                gLM_template = noisy_batch['template'].view(-1).to(noisy_batch['aatypes_t'].dtype)
                gLM_template[~noisy_batch['template_mask'].view(-1)] = noisy_batch['aatypes_t'].view(-1) # False in template_mask means the non-special tokens
                gLM_template = gLM_template.view(noisy_batch['aatypes_t'].shape[0], -1)
                plm_s_with_ST = self.folding_model.gLM_encoding(gLM_template)
                plm_s = plm_s_with_ST[~noisy_batch['template_mask']].view(*noisy_batch['aatypes_t'].shape, self.folding_model.plm_representations_layer+1, self.folding_model.plm_representations_dim)

            elif self.use_PLM.startswith('faESM2'):
                max_seqlen = noisy_batch['chain_lengthes'].max().item()
                cu_seqlens = torch.cumsum(noisy_batch['chain_lengthes'].view(-1), dim=0)
                cu_seqlens = torch.cat([torch.tensor([0]).int().to(cu_seqlens.device), cu_seqlens], dim=0)
                aatypes_in_ESM_templates = noisy_batch['template'].view(-1).to(noisy_batch['aatypes_t'].dtype)
                aatypes_in_ESM_templates[~noisy_batch['template_mask'].view(-1)] = noisy_batch['aatypes_t'].view(-1) # False in template_mask means the non-special tokens
                plm_s_with_ST = self.folding_model.faESM2_encoding(aatypes_in_ESM_templates.unsqueeze(0), cu_seqlens.int(), max_seqlen)
                plm_s = plm_s_with_ST[~noisy_batch['template_mask'].view(-1)].view(*noisy_batch['aatypes_t'].shape, self.folding_model.plm_representations_layer+1, self.folding_model.plm_representations_dim)
            
            else:
                raise NotImplementedError
            plm_s = plm_s.to(noisy_batch['trans_t'].dtype).detach()
            noisy_batch['PLM_embedding_aatypes_t'] = plm_s
            ### TODO: plm_z, the attention map, has not been intergated into the noisy_batch

        if curr_consistency_mode:
            # in cosistency mode, rollout will not be applied on the teacher model
            # so the t for any modality will be determined based on the orginal sampled t in the noisy_batch, instead rollout t
            rollout_time = self.d_t*curr_step_rollout_num # rollout_t = orgianl_t - rollout_time

            cat_t_adj = noisy_batch['cat_t'] + rollout_time + self.d_t
            cat_t_adj = cat_t_adj.clamp(min=self.model_start_t['backbone'], max=1)
            r3_t_adj = noisy_batch['r3_t'] + rollout_time + self.d_t
            r3_t_adj = r3_t_adj.clamp(min=self.model_start_t['backbone'], max=1)
            so3_t_adj = noisy_batch['so3_t'] + rollout_time + self.d_t
            so3_t_adj = so3_t_adj.clamp(min=self.model_start_t['backbone'], max=1)
            noisy_batch_t_adj = {'cat_t': cat_t_adj,
                                 'r3_t': r3_t_adj,
                                 'so3_t': so3_t_adj,}

            # aatypes_t_adj: seqeuence before cropped
            aatypes_t_adj, aatypes_sc_adj_t = self.interpolant.corrupt_aatypes_for_adj_t(aatypes_1=noisy_batch['aatypes_1'], 
                                                                                          cat_t=noisy_batch_t_adj['cat_t'], 
                                                                                          res_mask_aa=noisy_batch['res_mask_aa'], 
                                                                                          diffuse_mask_aa=noisy_batch['diffuse_mask_aa'],)
            # aatypes_t_adj: seqeuence before cropped
            noisy_batch_t_adj['aatypes_t'] = aatypes_t_adj
            noisy_batch_t_adj['aatypes_sc'] = aatypes_sc_adj_t

            if not self.use_PLM is None: 
                if self.use_PLM.startswith('gLM'):
                    gLM_template_t_adj = noisy_batch['template'].clone().view(-1).to(noisy_batch_t_adj['aatypes_t'].dtype)
                    gLM_template_t_adj[~noisy_batch['template_mask'].view(-1)] = noisy_batch_t_adj['aatypes_t'].view(-1) # False in template_mask means the non-special tokens
                    gLM_template_t_adj = gLM_template_t_adj.view(noisy_batch_t_adj['aatypes_t'].shape[0], -1)
                    plm_s_with_ST_t_adj = self.folding_model.gLM_encoding(gLM_template_t_adj)
                    plm_s_t_adj = plm_s_with_ST_t_adj[~noisy_batch['template_mask']].view(*noisy_batch_t_adj['aatypes_t'].shape, self.folding_model.plm_representations_layer+1, self.folding_model.plm_representations_dim)

                elif self.use_PLM.startswith('faESM2'):
                    aatypes_in_ESM_templates_t_adj = noisy_batch['template'].clone().view(-1).to(noisy_batch_t_adj['aatypes_t'].dtype)
                    aatypes_in_ESM_templates_t_adj[~noisy_batch['template_mask'].view(-1)] = noisy_batch_t_adj['aatypes_t'].view(-1) # False in template_mask means the non-special tokens
                    plm_s_with_ST_t_adj = self.folding_model.faESM2_encoding(aatypes_in_ESM_templates_t_adj.unsqueeze(0), cu_seqlens.int(), max_seqlen)
                    plm_s_t_adj = plm_s_with_ST_t_adj[~noisy_batch['template_mask'].view(-1)].view(*noisy_batch_t_adj['aatypes_t'].shape, self.folding_model.plm_representations_layer+1, self.folding_model.plm_representations_dim)
                
                else:
                    raise NotImplementedError
                plm_s_t_adj = plm_s_t_adj.to(noisy_batch['trans_t'].dtype).detach()
                noisy_batch_t_adj['PLM_embedding_aatypes_t'] = plm_s_t_adj
            
            # the remain terms have been cropped
            noisy_se3_t_adj = self.interpolant.corrupt_se3_for_adj_t(trans_1=noisy_batch['trans_1'], 
                                                                     r3_t=noisy_batch_t_adj['r3_t'], 
                                                                     rotmats_1=noisy_batch['rotmats_1'], 
                                                                     so3_t=noisy_batch_t_adj['so3_t'], 
                                                                     torsions_1=noisy_batch['torsions_1'], 
                                                                     res_mask=noisy_batch['res_mask'], 
                                                                     diffuse_mask=noisy_batch['diffuse_mask'], 
                                                                     rot_style='multiflow')
            for k, v in noisy_se3_t_adj.items():
                noisy_batch_t_adj[k] = v
            
        # crop input data
        batch_size, num_res = noisy_batch['aatypes_t'].shape
        if num_res > self._model_cfg.multimer_crop_threshold:
            noisy_batch['aatypes_1'] = noisy_batch['aatypes_1'][noisy_batch['crop_idx']].reshape(batch_size, self._model_cfg.multimer_crop_size)
            noisy_batch['aatypes_t'] = noisy_batch['aatypes_t'][noisy_batch['crop_idx']].reshape(batch_size, self._model_cfg.multimer_crop_size)
            noisy_batch['aatypes_sc'] = noisy_batch['aatypes_sc'][noisy_batch['crop_idx']].reshape(batch_size, self._model_cfg.multimer_crop_size, -1)
            noisy_batch['chain_idx'] = noisy_batch['chain_idx'][noisy_batch['crop_idx']].reshape(batch_size, self._model_cfg.multimer_crop_size)
            if self.use_PLM:
                _, _, last_dim_1, last_dim_0 = noisy_batch['PLM_embedding_aatypes_t'].shape
                noisy_batch['PLM_embedding_aatypes_t'] = noisy_batch['PLM_embedding_aatypes_t'][noisy_batch['crop_idx']].reshape(batch_size, self._model_cfg.multimer_crop_size, last_dim_1, last_dim_0)

            if curr_consistency_mode:
                noisy_batch_t_adj['aatypes_t'] = noisy_batch_t_adj['aatypes_t'][noisy_batch['crop_idx']].reshape(batch_size, self._model_cfg.multimer_crop_size)
                noisy_batch_t_adj['aatypes_sc'] = noisy_batch_t_adj['aatypes_sc'][noisy_batch['crop_idx']].reshape(batch_size, self._model_cfg.multimer_crop_size, -1)
                if self.use_PLM:
                    _, _, last_dim_1, last_dim_0 = noisy_batch_t_adj['PLM_embedding_aatypes_t'].shape
                    noisy_batch_t_adj['PLM_embedding_aatypes_t'] = noisy_batch_t_adj['PLM_embedding_aatypes_t'][noisy_batch['crop_idx']].reshape(batch_size, self._model_cfg.multimer_crop_size, last_dim_1, last_dim_0)
            
            del noisy_batch['crop_idx']
            del noisy_batch['res_mask_aa']
            del noisy_batch['diffuse_mask_aa']

        noisy_batch['logits_1'] = torch.nn.functional.one_hot(noisy_batch['aatypes_1'].long(), num_classes=self.aatype_pred_num_tokens).float()
        noisy_batch['PLM_emb_weight'] = self.folding_model.plm_emb_weight

        if curr_consistency_mode:
            noisy_batch_t_adj['res_mask'] = noisy_batch['res_mask']
            noisy_batch_t_adj['diffuse_mask'] = noisy_batch['diffuse_mask']
            noisy_batch_t_adj['res_idx'] = noisy_batch['res_idx']
            noisy_batch_t_adj['chain_idx'] = noisy_batch['chain_idx']
            noisy_batch_t_adj['interface_mask'] = noisy_batch['interface_mask']
            noisy_batch_t_adj['PLM_emb_weight'] = noisy_batch['PLM_emb_weight']

        ### Self-condition for the init step
        with torch.no_grad():
            if self._interpolant_cfg.self_condition >= random.random(): # self._interpolant_cfg.self_condition, a float between 0 and 1, representing the probability of self-conditioning
                if curr_step_rollout_num > 0: # if do rollout, the self-conditioning comes from the total models
                    noisy_batch = self.run_model(noisy_batch, until=None, mod_t=True)
                    last_model = self.model_order[-1]
                else: # otherwise, the self-conditioning comes from the current model
                    noisy_batch = self.run_model(noisy_batch, until=curr_step_training_models, mod_t=True)
                    last_model = curr_step_training_models[-1]
                
                if last_model == 'refine':
                    noisy_batch = self.update_self_condition(noisy_batch)
                else:
                    noisy_batch = self.update_self_condition_woRefine(noisy_batch)

                keys_to_remove = [k for k in noisy_batch if 'pred' in k] + \
                                 [k for k in noisy_batch if k.startswith('mod_')] + \
                                 ['backbone_init_node_embed', 'backbone_init_edge_embed', 'refine_model_mask']
                noisy_batch = {k:noisy_batch[k] for k in noisy_batch if k not in keys_to_remove}
                
                if curr_consistency_mode:
                    noisy_batch_t_adj['logits_1'] = noisy_batch['logits_1']
                    noisy_batch_t_adj['trans_1'] = noisy_batch['trans_1']
                    noisy_batch_t_adj['rotmats_1'] = noisy_batch['rotmats_1']
                    if curr_step_rollout_num > 0: # if do rollout, the self-conditioning comes from the total models
                        noisy_batch_t_adj = self.run_model(noisy_batch_t_adj, until=None, mod_t=True)
                    else: # otherwise, the self-conditioning comes from the current model
                        noisy_batch_t_adj = self.run_model(noisy_batch_t_adj, until=curr_step_training_models, mod_t=True)
                    noisy_batch_t_adj = self.update_self_condition(noisy_batch_t_adj)

                    keys_to_remove = [k for k in noisy_batch_t_adj if 'pred' in k] + \
                                    [k for k in noisy_batch_t_adj if k.startswith('mod_')] + \
                                    ['backbone_init_node_embed', 'backbone_init_edge_embed', 'refine_model_mask']
                    noisy_batch_t_adj = {k:noisy_batch_t_adj[k] for k in noisy_batch_t_adj if k not in keys_to_remove}
        
        # Rollout to simulate the inference process
        with torch.no_grad():
            for _ in range(curr_step_rollout_num):
                noisy_batch = self.run_model(noisy_batch, until=None, mod_t=True)
                invalid_ks = [i for i in noisy_batch if type(noisy_batch[i])==torch.Tensor and torch.any(torch.isnan(noisy_batch[i]))]
                if len(invalid_ks) > 0:
                    print(invalid_ks)
                
                noisy_batch = self.update_noised_data(noisy_batch, self.d_t)
                noisy_batch = self.update_self_condition(noisy_batch)
                
                keys_to_remove = [k for k in noisy_batch if 'pred' in k] + \
                                 [k for k in noisy_batch if k.startswith('mod_')] + \
                                 ['backbone_init_node_embed', 'backbone_init_edge_embed', 'refine_model_mask']
                noisy_batch = {k:noisy_batch[k] for k in noisy_batch if k not in keys_to_remove}
            # TODO: wheter a new self-condition from self.run_model(noisy_batch, until=curr_step_traininig_model, mod_t=True) is required
        
        # batch_losses, model_output = self.run_backbone_model_only(noisy_batch)
        batch_losses, model_output = self.model_step(noisy_batch, training_models=curr_step_training_models)

        if curr_consistency_mode:
            with torch.no_grad():
                model_output_adj_t = self.run_model(noisy_batch_t_adj, until=['backbone', ], mod_t=False)
                model_output_adj_t = {k:model_output_adj_t[k].detach() for k in ('backbone_pred_trans', 'backbone_pred_rotmats', 'backbone_pred_logits', 'cat_t', 'r3_t', 'so3_t')}
            loss_mask = noisy_batch['res_mask'] * noisy_batch['diffuse_mask']
            consistency_loss = self.cal_consistency_loss(model_output, model_output_adj_t, loss_mask=loss_mask)
            for k,v in consistency_loss.items():
                curr_consistency_loss_v = v * self._exp_cfg.consistency_loss_weight
                batch_losses[k] = curr_consistency_loss_v
                batch_losses['train_loss'] += curr_consistency_loss_v

        num_batch = batch_losses['train_loss'].shape[0]
        total_losses = {
            k: torch.mean(v) for k,v in batch_losses.items()
        }
        for k,v in total_losses.items():
            self._log_scalar(
                f"train/{k}", v, prog_bar=False, batch_size=num_batch)
        
        # Losses to track. Stratified across t.
        so3_t = torch.squeeze(noisy_batch['so3_t'])
        self._log_scalar(
            "train/so3_t",
            np.mean(du.to_numpy(so3_t)),
            prog_bar=False, batch_size=num_batch)
        r3_t = torch.squeeze(noisy_batch['r3_t'])
        self._log_scalar(
            "train/r3_t",
            np.mean(du.to_numpy(r3_t)),
            prog_bar=False, batch_size=num_batch)
        cat_t = torch.squeeze(noisy_batch['cat_t'])
        self._log_scalar(
            "train/cat_t",
            np.mean(du.to_numpy(cat_t)),
            prog_bar=False, batch_size=num_batch)
        tor_t = torch.squeeze(noisy_batch['tor_t'])
        self._log_scalar(
            "train/tor_t",
            np.mean(du.to_numpy(tor_t)),
            prog_bar=False, batch_size=num_batch)
        for loss_name, loss_dict in batch_losses.items():
            if loss_name.endswith('rotmats'):
                batch_t = so3_t
            elif loss_name.endswith('torsions'):
                batch_t = tor_t
            elif loss_name == 'train_loss':
                continue
            elif loss_name.endswith('aatypes'):
                batch_t = cat_t
            elif loss_name.endswith('trans'):
                batch_t = r3_t
            else:
                continue
            stratified_losses = mu.t_stratified_loss(
                batch_t, loss_dict, loss_name=loss_name)
            for k,v in stratified_losses.items():
                self._log_scalar(
                    f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # Training throughput
        self._log_scalar(
            "train/length", batch['res_mask'].shape[1], prog_bar=False, batch_size=num_batch)
        self._log_scalar(
            "train/batch_size", num_batch, prog_bar=False)
        step_time = time.time() - step_start_time
        self._log_scalar(
            "train/examples_per_second", num_batch / step_time)
        train_loss = total_losses['train_loss']
        self._log_scalar(
            "train/loss", train_loss, batch_size=num_batch)

        self.curr_training_step += 1

        return train_loss

    def configure_optimizers(self):
        
        param_group = []
        for model_name in self.model:
            if model_name == 'backbone':
                param_group.append({'params':self.model['backbone'].parameters(), 'lr':self._exp_cfg.optimizer.backbone_lr})
                self._print_logger.info(f'Set BackBone-Model LR: {self._exp_cfg.optimizer.backbone_lr}')
            elif model_name == 'sidechain':
                param_group.append({'params':self.model['sidechain'].parameters(), 'lr':self._exp_cfg.optimizer.sidechain_lr})
                self._print_logger.info(f'Set Sidechain-Model LR: {self._exp_cfg.optimizer.sidechain_lr}')
            elif model_name == 'refine':
                param_group.append({'params':self.model['refine'].parameters(), 'lr':self._exp_cfg.optimizer.refine_lr})
                self._print_logger.info(f'Set Refine-Model LR: {self._exp_cfg.optimizer.refine_lr}')

        return torch.optim.AdamW(params=param_group)

        # if self._exp_cfg.optimizer.type == 'adamw':
        #     return torch.optim.AdamW(
        #         params=full_params,
        #         lr=self._exp_cfg.optimizer.lr,
        #     )
        # elif self._exp_cfg.optimizer.type == 'adam':
        #     return torch.optim.Adam(
        #         params=full_params,
        #         lr=self._exp_cfg.optimizer.lr,
        #         betas=(self._exp_cfg.optimizer.beta1, self._exp_cfg.optimizer.beta2)
        #     )
        # else:
        #     raise ValueError(self._exp_cfg.optimizer.type)

    def set_seed(self):
        seed = self._infer_cfg.seed
        if self._infer_cfg.task.startswith('unconditional'):
            seed += dist.get_rank()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.seed_has_been_set = True

    def predict_step(self, batch, batch_idx):
        if not self.seed_has_been_set:
            self.set_seed()

        if self._infer_cfg.task == 'conditional':
            self.predict_conditional_step(batch, batch_idx)
            return

        if self._infer_cfg.task != 'unconditional_multimer_cbc':
            self.predict_backbone_step(batch, batch_idx)
        else:
            self.generate_multimer_chain_by_chain(batch, batch_idx)
        return 1

    def predict_backbone_step(self, batch, batch_idx):
        del batch_idx # Unused
        if 'res_mask_aa' in batch:
            del batch['res_mask_aa']
        if 'diffuse_mask_aa' in batch:
            del batch['diffuse_mask_aa']
        device = f'cuda:{torch.cuda.current_device()}'
        interpolant = Interpolant(self._infer_cfg.interpolant) 
        interpolant.set_device(device)

        if 'sample_id' in batch:
            sample_ids = batch['sample_id'].squeeze().tolist()
        else:
            sample_ids = [0]
        sample_ids = [sample_ids] if isinstance(sample_ids, int) else sample_ids
        num_batch = len(sample_ids)

        chain_idx = None
        res_idx = None
        true_full_atom_pos = None
        PLM_templates = None
        if self._infer_cfg.task.endswith('multimer'):
            chain_mode = 'multimer'
        else:
            chain_mode = 'monomer'

        PLM_embedder = self.folding_model
        PLM = PLM_embedder._cfg.PLM
        if not PLM_embedder.PLM_inited:
            if PLM.startswith('faESM'):
                PLM_embedder.init_faESM()
            elif PLM.startswith('gLM'):
                PLM_embedder.init_gLM()
            else:
                raise ValueError(f'PLM {PLM} not supported')
        plm_s = None

        if self._infer_cfg.task == 'unconditional':
            sample_length = batch['num_res'].item()
            true_bb_pos = None
            sample_dirs = [os.path.join(
                self.inference_dir, f'length_{sample_length}', f'sample_{str(sample_id)}')
                for sample_id in sample_ids]
            trans_1 = rotmats_1 = torsions_1 = diffuse_mask = aatypes_1 = true_aatypes = None

        elif self._infer_cfg.task == 'unconditional_multimer':

            N_max_res = int(batch['num_res'].max().item())
            N_total_res = int(batch['num_res'].sum().item())
            N_chains = batch['num_res'].shape[1] - 1 # the number of chains within the batch are the same
            N_batch = batch['num_res'].shape[0]
            if PLM.startswith('faESM'):
                N_max_res += 2
                template_mask = torch.zeros(N_total_res+N_batch*N_chains*2).bool().to(device)
                template = torch.zeros(N_total_res+N_batch*N_chains*2).long().to(device)
                num_res_with_special_token = batch['num_res'][:, 1:].view(-1) + 2
                cu_seqlens = torch.cat([torch.tensor([0]).to(num_res_with_special_token.device), num_res_with_special_token.cumsum(dim=0)])
                for chain_start_pos, chain_end_pos in zip(cu_seqlens[:-1], cu_seqlens[1:]):
                    template_mask[chain_start_pos.long()] = True
                    template[chain_start_pos.long()] = 21
                    template_mask[chain_end_pos.long()-1] = True
                    template[chain_end_pos.long()-1] = 22
            elif PLM.startswith('gLM'):
                N_max_res += 1
                template_mask = torch.zeros(N_total_res+N_batch*N_chains).bool().to(device)
                template = torch.zeros(N_total_res+N_batch*N_chains).long().to(device)
                num_res_with_special_token = batch['num_res'][:, 1:].view(-1) + 1
                cu_seqlens = torch.cat([torch.tensor([0]).to(num_res_with_special_token.device), num_res_with_special_token.cumsum(dim=0)])
                for chain_start_pos in cu_seqlens[:-1]:
                    template_mask[chain_start_pos.long()] = True
                    template[chain_start_pos.long()] = 21
            else:
                raise ValueError(f'PLM {PLM} not supported')

            PLM_templates = {}
            PLM_templates['N_max_res'] = N_max_res
            PLM_templates['template_mask'] = template_mask.view(N_batch, -1)
            PLM_templates['template'] = template
            PLM_templates['cu_seqlens'] = cu_seqlens

            merge_num_res = torch.Tensor(batch['num_res']).reshape(-1).to(torch.int64).to(batch['num_res'][0].device)
            batch['num_res'] = merge_num_res
            sample_length = batch['num_res'].sum().item()
            blk_length = batch['num_res']
            sample_length_str = '-'.join([str(blk) for blk in merge_num_res.cpu().tolist()])
            true_bb_pos = None
            sample_dirs = [os.path.join(
                self.inference_dir, f'length_{sample_length_str}', f'sample_{str(sample_id)}')
                for sample_id in sample_ids]
            trans_1 = rotmats_1 = torsions_1 = diffuse_mask = aatypes_1 = true_aatypes = None

            blk_indicator = blk_length.cumsum(dim=0)[:-1]
            chain_idx = torch.zeros(sample_length).to(torch.int64).to(batch['num_res'].device)
            chain_idx[blk_indicator] = 1
            chain_idx = chain_idx.cumsum(dim=0)[None, :].expand(num_batch, sample_length)

            res_idx = torch.arange(
                sample_length,
                device=batch['num_res'].device,
                dtype=torch.float32)
            res_idx_restart = torch.zeros_like(res_idx)
            N_chains = blk_length.shape[0]
            for chain_id in range(1, N_chains+1):
                chain_mask = chain_idx[0].eq(chain_id).int()
                chain_range_min_res_idx = (res_idx + (1-chain_mask) * 1e9).min()
                res_idx_restart += (res_idx - chain_range_min_res_idx) * chain_mask
            res_idx = res_idx_restart[None, :].expand(num_batch, sample_length)
            res_idx = res_idx + 1 # res_idx starts from 1
        
        elif self._infer_cfg.task == 'packing':
            sample_length = batch['trans_1'].shape[1]
            trans_1 = batch['trans_1']
            rotmats_1 = batch['rotmats_1']
            aatypes_1 = batch['aatypes_1']
            aatypes_1[aatypes_1>=self.aatype_pred_num_tokens] = 0
            true_torsions = batch['torsions_1']

            sample_dirs = [os.path.join(self.inference_dir, f'length_{sample_length}', batch['pdb_name'][0])]
            true_full_atom_pos = all_atom.atom37_from_trans_rot_torsion(trans_1, rotmats_1, true_torsions, aatypes_1, batch['res_mask'])
            assert true_full_atom_pos.shape == (1, sample_length, 37, 3)
            au.write_prot_to_pdb(
                prot_pos=true_full_atom_pos[0].cpu().detach().numpy(),
                file_path=os.path.join(sample_dirs[0], batch['pdb_name'][0] + '_gt.pdb'),
                aatype=batch['aatypes_1'][0].cpu().detach().numpy(),
            )
            torsions_1 = true_bb_pos = diffuse_mask = true_aatypes = None

        elif self._infer_cfg.task == 'forward_folding':
            sample_length = batch['trans_1'].shape[1]
            sample_dirs = [os.path.join(self.inference_dir, f'length_{sample_length}', batch['pdb_name'][0])]
            for sample_dir in sample_dirs:
                os.makedirs(sample_dir, exist_ok=True)
            true_bb_pos = all_atom.atom37_from_trans_rot(batch['trans_1'], batch['rotmats_1'])
            assert true_bb_pos.shape == (1, sample_length, 37, 3)
            # save the ground truth as a pdb
            au.write_prot_to_pdb(
                prot_pos=true_bb_pos[0].cpu().detach().numpy(),
                file_path=os.path.join(sample_dirs[0], batch['pdb_name'][0] + '_gt.pdb'),
                aatype=batch['aatypes_1'][0].cpu().detach().numpy(),
            )
            true_bb_pos = true_bb_pos[..., :3, :].reshape(-1, 3).cpu().numpy() 
            assert true_bb_pos.shape == (sample_length * 3, 3)
            aatypes_1 = batch['aatypes_1']
            aatypes_1[aatypes_1>=self.aatype_pred_num_tokens] = 0
            trans_1 = rotmats_1 = torsions_1= diffuse_mask = true_aatypes = None
            with torch.no_grad():
                if PLM == 'faESM2_650M':
                    max_seqlen = batch['chain_lengthes'].max().item()
                    cu_seqlens = torch.cumsum(batch['chain_lengthes'].view(-1), dim=0)
                    cu_seqlens = torch.cat([torch.tensor([0]).int().to(cu_seqlens.device), cu_seqlens], dim=0)
                    aatypes_in_ESM_templates = batch['template'].view(-1).to(batch['aatypes_1'].dtype)
                    aatypes_in_ESM_templates[~batch['template_mask'].view(-1)] = batch['aatypes_1'].view(-1) # False in template_mask means the non-special tokens
                    plm_s_with_ST = PLM_embedder.faESM2_encoding(aatypes_in_ESM_templates.unsqueeze(0), cu_seqlens.int(), max_seqlen)
                    plm_s = plm_s_with_ST[~batch['template_mask'].view(-1)].view(*batch['aatypes_1'].shape, PLM_embedder.plm_representations_layer+1, PLM_embedder.plm_representations_dim)
                elif PLM == 'gLM2_650M':
                    gLM_template = batch['template_mask'].view(-1).to(batch['aatypes_1'].dtype)
                    gLM_template[~batch['template_mask'].view(-1)] = batch['aatypes_1'].reshape(-1)
                    gLM_template = gLM_template.view(batch['aatypes_1'].shape[0], -1)
                    plm_s_with_ST = PLM_embedder.gLM_encoding(gLM_template)
                    plm_s = plm_s_with_ST[~batch['template_mask']].view(*batch['aatypes_1'].shape, PLM_embedder.plm_representations_layer+1, PLM_embedder.plm_representations_dim)
                else:
                    raise ValueError(PLM)

        elif self._infer_cfg.task == 'forward_folding_multimer':
            sample_length = batch['trans_1'].shape[1]
            sample_dirs = [os.path.join(self.inference_dir, f'length_{sample_length}', batch['pdb_name'][0])]
            for sample_dir in sample_dirs:
                os.makedirs(sample_dir, exist_ok=True)
            true_bb_pos = all_atom.atom37_from_trans_rot(batch['trans_1'], batch['rotmats_1'])
            assert true_bb_pos.shape == (1, sample_length, 37, 3)
            # save the ground truth as a pdb
            au.write_prot_to_pdb(
                prot_pos=true_bb_pos[0].cpu().detach().numpy(),
                file_path=os.path.join(sample_dirs[0], batch['pdb_name'][0] + '_gt.pdb'),
                aatype=batch['aatypes_1'][0].cpu().detach().numpy(),
                chain_index=batch['chain_idx'][0].cpu().detach().numpy(),
            )
            chain_idx = batch['chain_idx']
            true_bb_pos = true_bb_pos[..., :3, :].reshape(-1, 3).cpu().numpy() 
            assert true_bb_pos.shape == (sample_length * 3, 3)
            aatypes_1 = batch['aatypes_1']
            aatypes_1[aatypes_1>=self.aatype_pred_num_tokens] = 0
            trans_1 = rotmats_1 = torsions_1= diffuse_mask = true_aatypes = None
            with torch.no_grad():
                if PLM == 'faESM2_650M':
                    max_seqlen = batch['chain_lengthes'].max().item()
                    cu_seqlens = torch.cumsum(batch['chain_lengthes'].view(-1), dim=0)
                    cu_seqlens = torch.cat([torch.tensor([0]).int().to(cu_seqlens.device), cu_seqlens], dim=0)
                    aatypes_in_ESM_templates = batch['template'].view(-1).to(batch['aatypes_1'].dtype)
                    aatypes_in_ESM_templates[~batch['template_mask'].view(-1)] = batch['aatypes_1'].view(-1) # False in template_mask means the non-special tokens
                    plm_s_with_ST = PLM_embedder.faESM2_encoding(aatypes_in_ESM_templates.unsqueeze(0), cu_seqlens.int(), max_seqlen)
                    plm_s = plm_s_with_ST[~batch['template_mask'].view(-1)].view(*batch['aatypes_1'].shape, PLM_embedder.plm_representations_layer+1, PLM_embedder.plm_representations_dim)
                elif PLM == 'gLM2_650M':
                    gLM_template = batch['template_mask'].view(-1).to(batch['aatypes_1'].dtype)
                    gLM_template[~batch['template_mask'].view(-1)] = batch['aatypes_1'].reshape(-1)
                    gLM_template = gLM_template.view(batch['aatypes_1'].shape[0], -1)
                    plm_s_with_ST = PLM_embedder.gLM_encoding(gLM_template)
                    plm_s = plm_s_with_ST[~batch['template_mask']].view(*batch['aatypes_1'].shape, PLM_embedder.plm_representations_layer+1, PLM_embedder.plm_representations_dim)
                else:
                    raise ValueError(PLM)

        elif self._infer_cfg.task == 'inverse_folding':
            sample_length = batch['trans_1'].shape[1]
            trans_1 = batch['trans_1']
            rotmats_1 = batch['rotmats_1']
            torsions_1 = batch['torsions_1']
            true_aatypes = batch['aatypes_1']
            sample_dirs = [os.path.join(self.inference_dir, f'length_{sample_length}', batch['pdb_name'][0])]
            aatypes_1 = diffuse_mask = true_bb_pos = None

        elif self._infer_cfg.task == 'inverse_folding_multimer':
            sample_length = batch['trans_1'].shape[1]
            trans_1 = batch['trans_1']
            rotmats_1 = batch['rotmats_1']
            torsions_1 = batch['torsions_1']
            true_aatypes = batch['aatypes_1']
            sample_dirs = [os.path.join(self.inference_dir, f'length_{sample_length}', batch['pdb_name'][0])]
            aatypes_1 = diffuse_mask = true_bb_pos = None
            chain_idx = batch['chain_idx']

            PLM_templates = {}

            if PLM.startswith('faESM'):
                N_max_res = batch['chain_lengthes'].max().item()
                cu_seqlens = torch.cumsum(batch['chain_lengthes'].view(-1), dim=0)
                cu_seqlens = torch.cat([torch.tensor([0]).int().to(cu_seqlens.device), cu_seqlens], dim=0)
            elif PLM.startswith('gLM2'):
                N_max_res = None
                cu_seqlens = None
            else:
                raise ValueError(PLM)
            
            PLM_templates['N_max_res'] = N_max_res
            PLM_templates['template'] = batch['template'].view(-1).to(batch['aatypes_1'].dtype)
            PLM_templates['template_mask'] = batch['template_mask']
            PLM_templates['cu_seqlens'] = cu_seqlens


        else:
            raise ValueError(f'Unknown task {self._infer_cfg.task}')

        # Skip runs if already exist
        top_sample_csv_paths = [os.path.join(sample_dir, 'top_sample.csv')
                                for sample_dir in sample_dirs]
        if all([os.path.exists(top_sample_csv_path) for top_sample_csv_path in top_sample_csv_paths]):
            self._print_logger.info(f'Skipping instance {sample_ids} length {sample_length}')
            return

        # Sample batch
        prot_traj, model_traj, torsion_traj, model_torsion_traj, gt_result = interpolant.sample_backbone(
            num_batch, sample_length, 
            self.model,
            self.model_order,
            trans_1=trans_1, rotmats_1=rotmats_1, aatypes_1=aatypes_1, torsions_1=torsions_1, 
            diffuse_mask=diffuse_mask,
            chain_idx=chain_idx, res_idx=res_idx, 
            forward_folding=self._infer_cfg.task == 'forward_folding',
            inverse_folding=self._infer_cfg.task == 'inverse_folding',
            packing=self._infer_cfg.task == 'packing',
            separate_t=self._infer_cfg.interpolant.codesign_separate_t,
            rot_style=self._exp_cfg.rot_inference_style,
            PLM_embedder=PLM_embedder, 
            PLM_type=PLM, 
            PLM_encoding=plm_s,
            task=self._infer_cfg.task, 
            PLM_templates=PLM_templates,
        )
        diffuse_mask = diffuse_mask if diffuse_mask is not None else torch.ones(1, sample_length)
        atom37_traj = [x[0] for x in prot_traj]
        atom37_full_traj = [x[2] for x in prot_traj if len(x)==3]
        atom37_model_traj = [x[0] for x in model_traj]
        atom37_full_model_traj = [x[2] for x in model_traj if len(x)==3]

        bb_trajs = du.to_numpy(torch.stack(atom37_traj, dim=0).transpose(0, 1))
        # full_atom_trajs = du.to_numpy(torch.stack(atom37_full_traj, dim=0).transpose(0, 1))
        noisy_traj_length = bb_trajs.shape[1]
        assert bb_trajs.shape == (num_batch, noisy_traj_length, sample_length, 37, 3)
        if len(atom37_full_traj) > 1:
            full_atom_trajs = du.to_numpy(torch.stack(atom37_full_traj, dim=0).transpose(0, 1))
        else:
            full_atom_trajs = np.zeros((num_batch, 1, 1))
        # assert full_atom_trajs.shape == (num_batch, noisy_traj_length, sample_length, 37, 3)

        model_trajs = du.to_numpy(torch.stack(atom37_model_traj, dim=0).transpose(0, 1))
        clean_traj_length = model_trajs.shape[1]
        assert model_trajs.shape == (num_batch, clean_traj_length, sample_length, 37, 3)

        if len(atom37_full_model_traj) > 0:
            model_full_trajs = du.to_numpy(torch.stack(atom37_full_model_traj, dim=0).transpose(0, 1))
        else:
            model_full_trajs = np.zeros((num_batch, 1, 1))

        aa_traj = [x[1] for x in prot_traj]
        clean_aa_traj = [x[1] for x in model_traj]

        aa_trajs = du.to_numpy(torch.stack(aa_traj, dim=0).transpose(0, 1).long())
        assert aa_trajs.shape == (num_batch, noisy_traj_length, sample_length)

        for i in range(aa_trajs.shape[0]):
            for j in range(aa_trajs.shape[2]):
                if aa_trajs[i, -1, j] == du.MASK_TOKEN_INDEX:
                    print("WARNING mask in predicted AA")
                    aa_trajs[i, -1, j] = 0
        clean_aa_trajs = du.to_numpy(torch.stack(clean_aa_traj, dim=0).transpose(0, 1).long())
        assert clean_aa_trajs.shape == (num_batch, clean_traj_length, sample_length)

        for i, sample_id in zip(range(num_batch), sample_ids):
            sample_dir = sample_dirs[i]
            if chain_idx is None:
                sample_chain_index = None
            else:
                sample_chain_index = du.to_numpy(chain_idx[i])
            top_sample_df = self.compute_sample_metrics(
                batch,
                model_trajs[i],
                model_full_trajs[i],
                bb_trajs[i],
                full_atom_trajs[i],
                aa_trajs[i],
                clean_aa_trajs[i],
                true_bb_pos,
                true_aatypes,
                diffuse_mask,
                sample_id,
                sample_length,
                sample_dir,
                interpolant._aatypes_cfg.corrupt,
                self._infer_cfg.also_fold_pmpnn_seq,
                self._infer_cfg.write_sample_trajectories,
                chain_index=sample_chain_index,
                chain_mode=chain_mode,
            )
            top_sample_csv_path = os.path.join(sample_dir, 'top_sample.csv')
            top_sample_df.to_csv(top_sample_csv_path)

    def generate_multimer_chain_by_chain(self, batch, batch_idx):
        del batch_idx # Unused
        if 'res_mask_aa' in batch:
            del batch['res_mask_aa']
        if 'diffuse_mask_aa' in batch:
            del batch['diffuse_mask_aa']
        device = f'cuda:{torch.cuda.current_device()}'
        interpolant = Interpolant(self._infer_cfg.interpolant) 
        interpolant.set_device(device)

        if 'sample_id' in batch:
            sample_ids = batch['sample_id'].squeeze().tolist()
        else:
            sample_ids = [0]
        sample_ids = [sample_ids] if isinstance(sample_ids, int) else sample_ids
        num_batch = len(sample_ids)
        
        chain_mode = 'multimer'

        PLM_embedder = self.folding_model
        PLM = PLM_embedder._cfg.PLM
        if not PLM_embedder.PLM_inited:
            if PLM.startswith('faESM'):
                PLM_embedder.init_faESM()
            elif PLM.startswith('gLM'):
                PLM_embedder.init_gLM()
            else:
                raise ValueError(f'PLM {PLM} not supported')
        plm_s = None

        chain_lengths = batch['num_res'][0].cpu().tolist()[1:]
        chain_lengths = [int(cl) for cl in chain_lengths]
        sample_length = sum(chain_lengths)
        shuffled_chain_ids = list(range(1, len(chain_lengths)+1))
        random.shuffle(shuffled_chain_ids)
        accumulated_length = 0
        accumulated_chain_idx = None
        accumulated_res_idx = None
        pred_trans_1 = pred_rotmats_1 = pred_aatypes_1 = pred_torsions_1 = None

        sample_length_str = '-'.join([str(blk) for blk in chain_lengths])
        true_bb_pos = None
        true_aatypes = None
        sample_dirs = [os.path.join(
            self.inference_dir, f'length_{sample_length_str}', f'sample_{str(sample_id)}')
            for sample_id in sample_ids]

        for chain_idx, curr_chain_length in enumerate(chain_lengths):

            chain_id = shuffled_chain_ids[chain_idx]
            curr_diffuse_mask = torch.ones(num_batch, accumulated_length+curr_chain_length).int()
            curr_diffuse_mask[:, :accumulated_length] = 0
            curr_diffuse_mask = curr_diffuse_mask.to(device)

            new_chain_idx = chain_id * torch.ones(num_batch, curr_chain_length).to(torch.int64).to(batch['num_res'].device)
            new_res_idx = torch.arange(curr_chain_length,
                                       device=device,
                                       dtype=torch.float32)[None,:].expand(num_batch, curr_chain_length)

            curr_trans_0 = _centered_gaussian(num_batch, accumulated_length+curr_chain_length, self._device) * du.NM_TO_ANG_SCALE
            curr_rotmats_0 = interpolant.sample_rand_rot(num_batch, accumulated_length+curr_chain_length, dtype=torch.float)
            curr_aatypes_0 = _masked_categorical(num_batch, accumulated_length+curr_chain_length, self._device)
            ### if fixed sidechains, open this
            curr_torsions_0 = _uniform_torsion(num_batch, accumulated_length+curr_chain_length, self._device)
            # curr_torsions_0 = _uniform_torsion(num_batch, accumulated_length+curr_chain_length, self._device)

            if not accumulated_chain_idx is None:

                # shift the generated part, to leave the space center to the new chain
                # the first step is centerize the generated part
                pred_trans_1 = pred_trans_1 - pred_trans_1.mean(1, keepdim=True)
                # next shift the generated part
                intra_chain_dist = max(0, self._infer_cfg.shift_bias)
                binding_site_ = random.randint(int(pred_trans_1.shape[1]/3), int(2*pred_trans_1.shape[1]/3))
                residue_distance_to_center = torch.sqrt((pred_trans_1 ** 2).sum(-1)) # shape [B, L]
                _, binding_sites = residue_distance_to_center.topk(binding_site_)
                binding_site = binding_sites[:, -1]
                shift_direction = pred_trans_1[torch.arange(pred_trans_1.shape[0]).to(device), binding_site] # [B, 3]
                if intra_chain_dist > 0:
                    shift_distance = torch.sqrt((shift_direction ** 2).sum(1)) # the distance to shift the generated part, shape [B]
                    total_shift_distance = shift_distance + intra_chain_dist
                    shift_scale = total_shift_distance / shift_distance # [B,]
                    shift_direction = shift_direction * shift_scale[:, None]
                pred_trans_1 = pred_trans_1 - shift_direction[:, None, :]

                accumulated_chain_idx = torch.cat([accumulated_chain_idx, new_chain_idx], dim=1)
                accumulated_res_idx = torch.cat([accumulated_res_idx, new_res_idx], dim=1)
                # accumulated_trans_0 = torch.cat([pred_trans_1, curr_trans_0], dim=1)
                # accumulated_rotmats_0 = torch.cat([pred_rotmats_1, curr_rotmats_0], dim=1)
                # accumulated_aatypes_0 = torch.cat([pred_aatypes_1, curr_aatypes_0], dim=1)
                ### if fixed sidechains, open this
                # accumulated_torsions_0 = torch.cat([pred_torsions_1, curr_torsions_0], dim=1)
                # accumulated_torsions_0 = curr_torsions_0

                accumulated_trans_0 = curr_trans_0
                accumulated_rotmats_0 = curr_rotmats_0
                accumulated_aatypes_0 = curr_aatypes_0
                accumulated_torsions_0 = curr_torsions_0

            else:
                accumulated_chain_idx = new_chain_idx
                accumulated_res_idx = new_res_idx
                accumulated_trans_0 = curr_trans_0
                accumulated_rotmats_0 = curr_rotmats_0
                accumulated_aatypes_0 = curr_aatypes_0
                accumulated_torsions_0 = curr_torsions_0
            
            accumulated_aatypes_0 = accumulated_aatypes_0.int()

            accumulated_length += curr_chain_length

            curr_N_total_res = sum(chain_lengths[:chain_idx+1])
            curr_N_chains = chain_idx+1
            curr_N_max_res = max(chain_lengths[:chain_idx+1])

            if PLM.startswith('faESM'):
                curr_N_max_res += 2
                template_mask = torch.zeros(curr_N_total_res+num_batch*curr_N_chains*2).bool().to(device)
                template = torch.zeros(curr_N_total_res+num_batch*curr_N_chains*2).long().to(device)
                num_res_with_special_token = torch.Tensor([cl+2 for cl in chain_lengths[:chain_idx+1]] * num_batch).to(device).to(batch['num_res'].dtype)
                cu_seqlens = torch.cat([torch.tensor([0]).to(num_res_with_special_token.device), num_res_with_special_token.cumsum(dim=0)])
                for chain_start_pos, chain_end_pos in zip(cu_seqlens[:-1], cu_seqlens[1:]):
                    template_mask[chain_start_pos.long()] = True
                    template[chain_start_pos.long()] = 21
                    template_mask[chain_end_pos.long()-1] = True
                    template[chain_end_pos.long()-1] = 22
            elif PLM.startswith('gLM'):
                curr_N_max_res += 1
                template_mask = torch.zeros(curr_N_total_res+num_batch*curr_N_chains).bool().to(device)
                template = torch.zeros(curr_N_total_res+num_batch*curr_N_chains).long().to(device)
                num_res_with_special_token = torch.Tensor([cl+1 for cl in chain_lengths[:chain_idx+1]] * num_batch).to(device).to(batch['num_res'].dtype)
                cu_seqlens = torch.cat([torch.tensor([0]).to(num_res_with_special_token.device), num_res_with_special_token.cumsum(dim=0)])
                for chain_start_pos in cu_seqlens[:-1]:
                    template_mask[chain_start_pos.long()] = True
                    template[chain_start_pos.long()] = 21
            else:
                raise ValueError(f'PLM {PLM} not supported')

            PLM_templates = {}
            PLM_templates['N_max_res'] = curr_N_max_res
            PLM_templates['template_mask'] = template_mask.view(num_batch, -1)
            PLM_templates['template'] = template
            PLM_templates['cu_seqlens'] = cu_seqlens

            prot_traj, model_traj, torsion_traj, model_torsion_traj, gt_result, pred = interpolant.sample_chain_by_chain(
                num_batch, accumulated_length, 
                self.model,
                self.model_order,
                trans_1=pred_trans_1, rotmats_1=pred_rotmats_1, aatypes_1=pred_aatypes_1, torsions_1=accumulated_torsions_0, 
                trans_0=accumulated_trans_0, rotmats_0=accumulated_rotmats_0, aatypes_0=accumulated_aatypes_0, torsions_0=accumulated_torsions_0,
                diffuse_mask_gen=curr_diffuse_mask,
                chain_idx=accumulated_chain_idx, res_idx=accumulated_res_idx, 
                rot_style=self._exp_cfg.rot_inference_style,
                PLM_embedder=PLM_embedder, 
                PLM_type=PLM, 
                PLM_encoding=plm_s,
                PLM_templates=PLM_templates,
            )

            if len(pred) == 3:
                pred_trans_1, pred_rotmats_1, pred_aatypes_1 = pred
            else:
                pred_trans_1, pred_rotmats_1, pred_aatypes_1, pred_torsions_1 = pred

        diffuse_mask = torch.ones_like(curr_diffuse_mask)
        atom37_traj = [x[0] for x in prot_traj]
        atom37_full_traj = [x[2] for x in prot_traj if len(x)==3]
        atom37_model_traj = [x[0] for x in model_traj]
        atom37_full_model_traj = [x[2] for x in model_traj if len(x)==3]

        bb_trajs = du.to_numpy(torch.stack(atom37_traj, dim=0).transpose(0, 1))
        # full_atom_trajs = du.to_numpy(torch.stack(atom37_full_traj, dim=0).transpose(0, 1))
        noisy_traj_length = bb_trajs.shape[1]
        assert bb_trajs.shape == (num_batch, noisy_traj_length, sample_length, 37, 3)
        if len(atom37_full_traj) > 1:
            full_atom_trajs = du.to_numpy(torch.stack(atom37_full_traj, dim=0).transpose(0, 1))
        else:
            full_atom_trajs = np.zeros((num_batch, 1, 1))
        # assert full_atom_trajs.shape == (num_batch, noisy_traj_length, sample_length, 37, 3)

        model_trajs = du.to_numpy(torch.stack(atom37_model_traj, dim=0).transpose(0, 1))
        clean_traj_length = model_trajs.shape[1]
        assert model_trajs.shape == (num_batch, clean_traj_length, sample_length, 37, 3)

        if len(atom37_full_model_traj) > 0:
            model_full_trajs = du.to_numpy(torch.stack(atom37_full_model_traj, dim=0).transpose(0, 1))
        else:
            model_full_trajs = np.zeros((num_batch, 1, 1))

        aa_traj = [x[1] for x in prot_traj]
        clean_aa_traj = [x[1] for x in model_traj]

        aa_trajs = du.to_numpy(torch.stack(aa_traj, dim=0).transpose(0, 1).long())
        assert aa_trajs.shape == (num_batch, noisy_traj_length, sample_length)

        for i in range(aa_trajs.shape[0]):
            for j in range(aa_trajs.shape[2]):
                if aa_trajs[i, -1, j] == du.MASK_TOKEN_INDEX:
                    print("WARNING mask in predicted AA")
                    aa_trajs[i, -1, j] = 0
        clean_aa_trajs = du.to_numpy(torch.stack(clean_aa_traj, dim=0).transpose(0, 1).long())
        assert clean_aa_trajs.shape == (num_batch, clean_traj_length, sample_length)

        for i, sample_id in zip(range(num_batch), sample_ids):
            sample_dir = sample_dirs[i]
            sample_chain_index = du.to_numpy(accumulated_chain_idx[i])
            top_sample_df = self.compute_sample_metrics(
                batch,
                model_trajs[i],
                model_full_trajs[i],
                bb_trajs[i],
                full_atom_trajs[i],
                aa_trajs[i],
                clean_aa_trajs[i],
                true_bb_pos,
                true_aatypes,
                diffuse_mask,
                sample_id,
                sample_length,
                sample_dir,
                interpolant._aatypes_cfg.corrupt,
                self._infer_cfg.also_fold_pmpnn_seq,
                self._infer_cfg.write_sample_trajectories,
                chain_index=sample_chain_index,
                chain_mode=chain_mode,
            )
            top_sample_csv_path = os.path.join(sample_dir, 'top_sample.csv')
            top_sample_df.to_csv(top_sample_csv_path)

    def predict_conditional_step(self, batch, batch_idx):
        del batch_idx # Unused

        device = f'cuda:{torch.cuda.current_device()}'
        interpolant = Interpolant(self._infer_cfg.interpolant) 
        interpolant.set_device(device)

        PLM_embedder = self.folding_model
        PLM = PLM_embedder._cfg.PLM
        if not PLM_embedder.PLM_inited:
            if PLM.startswith('faESM'):
                PLM_embedder.init_faESM()
            elif PLM.startswith('gLM'):
                PLM_embedder.init_gLM()
            else:
                raise ValueError(f'PLM {PLM} not supported')

        num_res = batch['trans_1'].shape[1]
        num_batch = batch['trans_1'].shape[0]

        plm_s = None

        PLM_templates = {}
        PLM_templates['gt_aatypes_1'] = copy.deepcopy(batch['aatypes_1'].long())
        PLM_templates['gt_chain_idx'] = copy.deepcopy(batch['chain_idx'])
        PLM_templates['gt_diffuse_mask'] = copy.deepcopy(batch['diffuse_mask'])
        PLM_templates['gt_res_mask'] = copy.deepcopy(batch['res_mask'])

        PLM_templates['aatypes_1_design'] = copy.deepcopy(batch['aatypes_1'][:, :self._eval_dataset.sample_length].long())
        PLM_templates['chain_idx_design'] = copy.deepcopy(batch['chain_idx'][:, :self._eval_dataset.sample_length])
        PLM_templates['diffuse_mask_design'] = copy.deepcopy(batch['diffuse_mask'][:, :self._eval_dataset.sample_length])
        PLM_templates['res_mask_design'] = copy.deepcopy(batch['res_mask'][:, :self._eval_dataset.sample_length])

        if PLM.startswith('faESM'):
            PLM_templates['template_design'] = copy.deepcopy(batch['template'][:, :self._eval_dataset.sample_length+2].contiguous()) # bos and eos
            PLM_templates['template_mask_design'] = copy.deepcopy(batch['template_mask'][:, :self._eval_dataset.sample_length+2].contiguous()) # bos and eos
            PLM_templates['chain_lengthes_design'] = torch.tensor([self._eval_dataset.sample_length+2]).to(batch['res_mask'].device)[None, ...].repeat_interleave(num_batch, dim=0)
        elif PLM.startswith('gLM'):
            PLM_templates['template_design'] = copy.deepcopy(batch['template'][:, :self._eval_dataset.sample_length+1].contiguous()) # <+>
            PLM_templates['template_mask_design'] = copy.deepcopy(batch['template_mask'][:, :self._eval_dataset.sample_length+1].contiguous()) # <+>
        else:
            raise ValueError(f'PLM {PLM} not supported')

        sampled_trans_1 = batch['trans_1']
        sampled_rotmats_1 = batch['rotmats_1']
        sampled_aatypes_1 = batch['aatypes_1'].long()
        sampled_torsions_1 = batch['torsions_1']
        sampled_diffuse_mask = batch['diffuse_mask']
        sampled_res_mask = batch['res_mask']
        sampled_chain_idx = batch['chain_idx']
        sampled_res_idx = batch['res_idx']

        # run PLM with full seqs
        with torch.no_grad():
            # PLM_templates['gt_aatypes_1'][PLM_templates['gt_diffuse_mask_aa'].bool()] = sampled_aatypes_0[sampled_diffuse_mask.bool()].long()
            if PLM.startswith('faESM'):
                # PLM_templates['gt_aatypes_1'] = torch.where(PLM_templates['gt_res_mask'].bool(), PLM_templates['gt_aatypes_1'], torch.ones_like(PLM_templates['gt_aatypes_1'])*23)
                max_seqlen = batch['chain_lengthes'].max().item()
                cu_seqlens = torch.cumsum(batch['chain_lengthes'].view(-1), dim=0)
                cu_seqlens = torch.cat([torch.tensor([0]).int().to(cu_seqlens.device), cu_seqlens], dim=0)
                aatypes_in_ESM_templates = batch['template'].view(-1).to(PLM_templates['gt_aatypes_1'].dtype)
                aatypes_in_ESM_templates[~batch['template_mask'].view(-1)] = PLM_templates['gt_aatypes_1'].view(-1) # False in template_mask means the non-special tokens
                plm_s_with_ST = PLM_embedder.faESM2_encoding(aatypes_in_ESM_templates.unsqueeze(0), cu_seqlens.int(), max_seqlen)
                plm_s = plm_s_with_ST[~batch['template_mask'].view(-1)].view(*PLM_templates['gt_aatypes_1'].shape, PLM_embedder.plm_representations_layer+1, PLM_embedder.plm_representations_dim)
            elif PLM.startswith('gLM'):
                PLM_templates['gt_aatypes_1'] = torch.where(PLM_templates['gt_res_mask_aa'].bool(), PLM_templates['gt_aatypes_1'], torch.ones_like(PLM_templates['gt_aatypes_1'])*22)
                gLM_template = batch['template'].view(-1).to(PLM_templates['gt_aatypes_1'].dtype)
                gLM_template[~batch['template_mask'].view(-1)] = PLM_templates['gt_aatypes_1'].view(-1) # False in template_mask means the non-special tokens
                gLM_template = gLM_template.view(PLM_templates['gt_aatypes_1'].shape[0], -1)
                plm_s_with_ST = self.folding_model.gLM_encoding(gLM_template)
                plm_s = plm_s_with_ST[~batch['template_mask']].view(*PLM_templates['gt_aatypes_1'].shape, PLM_embedder.plm_representations_layer+1, PLM_embedder.plm_representations_dim)

        plm_s = plm_s.to(batch['trans_1'].dtype)
        PLM_templates['plm_s'] = plm_s

        # Sample batch
        prot_traj, _, _, _, gt_result = interpolant.sample_conditional(
            num_batch, num_res,
            self.model,
            self.model_order,
            trans_1=sampled_trans_1, rotmats_1=sampled_rotmats_1, aatypes_1=sampled_aatypes_1, torsions_1=sampled_torsions_1,
            trans_0=sampled_trans_1, rotmats_0=sampled_rotmats_1, aatypes_0=sampled_aatypes_1, torsions_0=sampled_torsions_1,
            diffuse_mask=sampled_diffuse_mask,
            separate_t=self._infer_cfg.interpolant.codesign_separate_t,
            chain_idx=sampled_chain_idx,
            res_idx=sampled_res_idx,
            res_mask=sampled_res_mask,
            PLM_embedder=PLM_embedder,
            PLM_type=PLM,
            PLM_templates=PLM_templates,

        )

        atom37_traj = [x[0] for x in prot_traj] # bb
        atom37_full_traj = [x[2] for x in prot_traj if len(x)==3] # all atom

        bb_trajs = du.to_numpy(torch.stack(atom37_traj, dim=0).transpose(0, 1))
        full_atom_trajs = du.to_numpy(torch.stack(atom37_full_traj, dim=0).transpose(0, 1))
        noisy_traj_length = bb_trajs.shape[1]
        assert bb_trajs.shape == (num_batch, noisy_traj_length, num_res, 37, 3)

        aa_traj = [x[1] for x in prot_traj]

        aa_trajs = du.to_numpy(torch.stack(aa_traj, dim=0).transpose(0, 1).long())
        assert aa_trajs.shape == (num_batch, noisy_traj_length, num_res)

        for i in range(aa_trajs.shape[0]):
            for j in range(aa_trajs.shape[2]):
                if aa_trajs[i, -1, j] == du.MASK_TOKEN_INDEX:
                    print("WARNING mask in predicted AA")
                    aa_trajs[i, -1, j] = 0
                    
        pdb_name = self._eval_dataset.pdb_name
        if not os.path.exists(self.inference_dir):
            os.makedirs(self.inference_dir)

        for i in range(num_batch):
            if len(self.model_order) == 1:
                prot_pos = bb_trajs[i][-1]
            else:
                prot_pos = full_atom_trajs[i][-1] # [seq, 37, 3]

            aatypes = aa_trajs[i][-1]

            pdb_path = os.path.join(self.inference_dir, f"{pdb_name}_{batch['idx'][i].cpu().item()}.pdb")

            self._print_logger.info(f'Writing {pdb_path}')

            au.write_prot_to_pdb(
                prot_pos=prot_pos,
                file_path=pdb_path,
                aatype=aatypes,
                no_indexing=True,
                chain_index=sampled_chain_idx.cpu()[i].numpy(),
            )


    def run_pmpnn(
            self,
            write_dir,
            pdb_input_path,
            chain_mode='monomer',
        ):
        if chain_mode == 'monomer':
            self.folding_model.run_pmpnn(
                write_dir,
                pdb_input_path,
            )
        else:
            self.folding_model.run_pmpnn_multimer(
                write_dir,
                pdb_input_path,
            )
        mpnn_fasta_path = os.path.join(
            write_dir,
            'seqs',
            os.path.basename(pdb_input_path).replace('.pdb', '.fa')
        )
        fasta_seqs = fasta.FastaFile.read(mpnn_fasta_path)
        all_header_seqs = [
            (f'pmpnn_seq_{i}', seq) for i, (_, seq) in enumerate(fasta_seqs.items())
            if i > 0
        ]
        modified_fasta_path = mpnn_fasta_path.replace('.fa', '_modified.fasta')
        fasta.FastaFile.write_iter(modified_fasta_path, all_header_seqs)
        return modified_fasta_path


    def compute_sample_metrics(self, batch, 
                               model_traj, model_full_traj, 
                               bb_traj, full_atom_traj, 
                               aa_traj, clean_aa_traj, 
                               true_bb_pos, true_aa, diffuse_mask,
                               sample_id, sample_length, sample_dir,
                               aatypes_corrupt,
                               also_fold_pmpnn_seq, write_sample_trajectories,
                               chain_index=None, chain_mode='monomer'):

        noisy_traj_length, sample_length, _, _ = bb_traj.shape
        clean_traj_length = model_traj.shape[0]
        assert bb_traj.shape == (noisy_traj_length, sample_length, 37, 3)
        assert model_traj.shape == (clean_traj_length, sample_length, 37, 3)
        assert aa_traj.shape == (noisy_traj_length, sample_length)
        assert clean_aa_traj.shape == (clean_traj_length, sample_length)


        os.makedirs(sample_dir, exist_ok=True)

        postfix = '_'+'_'.join(sample_dir.split('/')[-2:])

        traj_paths = eu.save_traj(
            bb_traj[-1],
            full_atom_traj[-1],
            bb_traj,
            full_atom_traj,
            np.flip(model_traj, axis=0),
            np.flip(model_full_traj, axis=0),
            du.to_numpy(diffuse_mask)[0],
            output_dir=sample_dir,
            aa_traj=aa_traj, 
            clean_aa_traj = clean_aa_traj,
            chain_index = chain_index,
            write_trajectories=write_sample_trajectories,
            postfix = '_'+'_'.join(sample_dir.split('/')[-2:])
        )

        # Run PMPNN to get sequences
        sc_output_dir = os.path.join(sample_dir, 'self_consistency')
        os.makedirs(sc_output_dir, exist_ok=True)
        pdb_path = traj_paths['sample_path']
        pmpnn_pdb_path = os.path.join(
            sc_output_dir, os.path.basename(pdb_path))
        shutil.copy(pdb_path, pmpnn_pdb_path)
        assert (diffuse_mask == 1.0).all()
        pmpnn_fasta_path = self.run_pmpnn(
            sc_output_dir,
            pmpnn_pdb_path,
            chain_mode,
        )

        os.makedirs(os.path.join(sc_output_dir, 'codesign_seqs'), exist_ok=True)
        codesign_fasta = fasta.FastaFile()
        codesign_fasta['codesign_seq_1'] = "".join([restypes[x] for x in aa_traj[-1]])
        codesign_fasta_path = os.path.join(sc_output_dir, 'codesign_seqs', 'codesign.fa')
        codesign_fasta.write(codesign_fasta_path)


        folded_dir = os.path.join(sc_output_dir, 'folded')
        if os.path.exists(folded_dir):
            shutil.rmtree(folded_dir)
        os.makedirs(folded_dir, exist_ok=False)
        if aatypes_corrupt:
            # codesign metrics
            folded_output = self.folding_model.fold_fasta(codesign_fasta_path, folded_dir)

            if also_fold_pmpnn_seq:
                pmpnn_folded_output = self.folding_model.fold_fasta(pmpnn_fasta_path, folded_dir)
                pmpnn_results = mu.process_folded_outputs(pdb_path, pmpnn_folded_output, true_bb_pos)
                pmpnn_results.to_csv(os.path.join(sample_dir, 'pmpnn_results.csv'))

        else:
            # non-codesign metrics
            folded_output = self.folding_model.fold_fasta(pmpnn_fasta_path, folded_dir)

        mpnn_results = mu.process_folded_outputs(pdb_path, folded_output, true_bb_pos)
        # true_bb_pos is not None only when executing forward folding task

        if true_aa is not None:
            assert true_aa.shape == (1, sample_length)

            true_aa_fasta = fasta.FastaFile()
            true_aa_fasta['seq_1'] = "".join([restypes_with_x[i] for i in true_aa[0]])
            true_aa_fasta.write(os.path.join(sample_dir, 'true_aa.fa'))

            seq_recovery = (torch.from_numpy(aa_traj[-1]).to(true_aa[0].device) == true_aa[0]).float().mean()
            mpnn_results['inv_fold_seq_recovery'] = seq_recovery.item()

            # get seq recovery for PMPNN as well
            if also_fold_pmpnn_seq:
                pmpnn_fasta = fasta.FastaFile.read(pmpnn_fasta_path)
                pmpnn_fasta_str = pmpnn_fasta['pmpnn_seq_1']
                pmpnn_fasta_idx = torch.tensor([restypes_with_x.index(x) for x in pmpnn_fasta_str if not x=='/']).to(true_aa[0].device)
                pmpnn_seq_recovery = (pmpnn_fasta_idx == true_aa[0]).float().mean()
                pmpnn_results['pmpnn_seq_recovery'] = pmpnn_seq_recovery.item()
                pmpnn_results.to_csv(os.path.join(sample_dir, 'pmpnn_results.csv'))
                mpnn_results['pmpnn_seq_recovery'] = pmpnn_seq_recovery.item()
                mpnn_results['pmpnn_bb_rmsd'] = pmpnn_results['bb_rmsd']

        # Save results to CSV
        mpnn_results.to_csv(os.path.join(sample_dir, 'sc_results.csv'))
        mpnn_results['length'] = sample_length
        mpnn_results['sample_id'] = sample_id
        del mpnn_results['header']
        del mpnn_results['sequence']

        # Select the top sample
        top_sample = mpnn_results.sort_values('bb_rmsd', ascending=True).iloc[:1]

        # Compute secondary structure metrics
        sample_dict = top_sample.iloc[0].to_dict()
        ss_metrics = mu.calc_mdtraj_metrics(sample_dict['sample_path'])
        top_sample['helix_percent'] = ss_metrics['helix_percent']
        top_sample['strand_percent'] = ss_metrics['strand_percent']
        return top_sample