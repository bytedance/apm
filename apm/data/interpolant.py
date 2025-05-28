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
import copy
import math
import numpy as np
import functools as fn
import torch.nn.functional as F
from functools import partial
from collections import defaultdict
from apm.data import so3_utils, all_atom, so2_utils
from apm.data import utils as du
from apm.analysis import utils as au
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
from torch import autograd
from torch.distributions.categorical import Categorical
from torch.distributions.binomial import Binomial
import random
import tempfile
from io import StringIO

# from apm.data.foldflow.rot_sampling import reverse as foldflow_reverse
# from apm.data.foldflow.rot_operator import vectorfield
# from apm.data.foldflow.so3_helpers import so3_relative_angle

from apm.models.utils import separate_multimer, merge_multimer


def exp_schedule(t, max_temp, decay_rate):
    return max_temp * np.exp(-decay_rate * t)


def _centered_gaussian(num_batch, num_res, device):
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)


def _uniform_so3(num_batch, num_res, device):
    return torch.tensor(
        Rotation.random(num_batch*num_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)

def _uniform_torsion(num_batch, num_res, device):
    return torch.rand(size=(num_batch, num_res, 4), device=device, dtype=torch.float32) * 2 * np.pi

def _aatype_prior_torsion(aatypes, device):
    return None

def _masked_categorical(num_batch, num_res, device):
    return torch.ones(
        num_batch, num_res, device=device) * du.MASK_TOKEN_INDEX


def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])


def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return (
        rotmats_t * diffuse_mask[..., None, None]
        + rotmats_1 * (1 - diffuse_mask[..., None, None])
    )

def _torsions_diffuse_mask(torsions_t, torsions_1, diffuse_mask):
    return (
        torsions_t * diffuse_mask[..., None]
        + torsions_1 * (1 - diffuse_mask[..., None])
    )

def _aatypes_diffuse_mask(aatypes_t, aatypes_1, diffuse_mask):
    return aatypes_t * diffuse_mask + aatypes_1 * (1 - diffuse_mask)


class Interpolant:

    def __init__(self, cfg):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._aatypes_cfg = cfg.aatypes
        self._torsions_cfg = cfg.torsions
        self._sample_cfg = cfg.sampling
        self._igso3 = None
        self._ot_fn = 'exact'

        self.num_tokens = 21 if self._aatypes_cfg.interpolant_type == "masking" else 20
        assert self._cfg.refine_start_t >= self._cfg.sidechain_start_t

        self.task_prob = {'folding':self._cfg.codesign_forward_fold_prop,
                          'inverse-folding':self._cfg.codesign_inverse_fold_prop, 
                          'packing':self._cfg.codesign_packing_prop, 
                          'codesign':self._cfg.codesign_prop,}

        self.start_t = {'backbone':self._cfg.min_t, 
                        'sidechain':self._cfg.sidechain_start_t, 
                        'refine':self._cfg.refine_start_t}
        self.end_t = 1 - self._cfg.min_t

        self.data_modality = {'backbone':(['aatypes', ], ['trans', 'rotmats', ]), 
                              'sidechain':(['aatypes', 'trans', 'rotmats', ], ['torsions', ], ), 
                              'refine':(['aatypes', ], ['trans', 'rotmats', ], ['torsions', ], ), 
                              'sidechain-refine':(['aatypes', ], ['trans', 'rotmats', ], ['torsions', ], ), }

        self.model_support_tasks = {'backbone':('codesign', 'folding', 'inverse-folding'),
                                    'sidechain':('codesign', 'packing', ), 
                                    # for sidechain model, 
                                    # codesign represents packing in codesign process and requires predicted structure & aatype as input to simulate the sampling process in full-atom generation
                                    # packing represents the exactly packing task, which requires ground truth structure & aatype as input
                                    'refine':('codesign', 'folding', 'inverse-folding'), 
                                    'sidechain-refine':('codesign', 'folding', 'inverse-folding', 'packing'),}

        self.t_mapping = {'aatypes':'cat_t', 
                          'trans':'r3_t', 
                          'rotmats':'so3_t',
                          'torsions':'tor_t', }
        
        assert self._aatypes_cfg.interpolant_type == "masking"

        # from https://github.com/jasonkyuyim/multiflow/blob/main/multiflow/models/utils.py#L269
        self.reference_normalized_counts = torch.tensor([
            0.0739, 0.05378621, 0.0410424, 0.05732177, 0.01418736, 0.03995128,
            0.07562267, 0.06695857, 0.02163064, 0.0580802, 0.09333149, 0.06777057,
            0.02034217, 0.03673995, 0.04428474, 0.05987899, 0.05502958, 0.01228988,
            0.03233601, 0.07551553
        ])

    def norm_task_prob(self, excute_tasks):
        excute_task_prob = {task_:self.task_prob[task_] for task_ in excute_tasks}
        if 'codesign' in excute_tasks and self.task_prob['codesign'] == -1:
            excute_task_prob['codesign'] = 1 - sum([excute_task_prob[task_] for task_ in excute_tasks if task_!='codesign'])
        sum_prob = sum([excute_task_prob[task_] for task_ in excute_tasks])
        for task_ in excute_tasks:
            excute_task_prob[task_] = excute_task_prob[task_] / sum_prob
        return excute_task_prob

    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(
                1000, sigma_grid, cache_dir='.cache')
        return self._igso3

    def sample_rand_rot(self, N_batch, N_res, dtype):
        sample_rotmats_0 = Rotation.random(N_batch*N_res).as_matrix()
        sample_rotmats_0 = torch.from_numpy(sample_rotmats_0).to(self._device, dtype=dtype)
        sample_rotmats_0 = sample_rotmats_0.reshape(N_batch, N_res, 3, 3)
        return sample_rotmats_0

    def set_device(self, device):
        self._device = device
    
    def sample_t_with_range(self, num_batch, min_t, max_t):
        if type(min_t) is int:
            min_t = min_t * torch.ones(num_batch, device=self._device) # [B, ]
        if type(max_t) is int:
            max_t = max_t * torch.ones(num_batch, device=self._device) # [B, ]
        t_range = max_t - min_t # [B, ]
        t = torch.rand(num_batch, device=self._device) # [B, ]
        return t * t_range + min_t
    
    def sample_pm(self, num_batch, batch_length, mask_ratio=0.15):
        # randomly sample the mask residues in partial co-design task
        # if mask_ratio is positive float, all residues in all batches will be masked with the same probability, mask_ratio
        # else if mask_ratio is -1, each samples will be applied a random mask_ratio 
 
        assert (mask_ratio > 0 and mask_ratio<1) or mask_ratio == -1, f"wrong codesign_partial_ratio: {mask_ratio}, should belongs to (0,1) or equal to -1"

        u = torch.rand((num_batch, batch_length), device=self._device)
        if mask_ratio > 0:
            diffuse_mask = (u < mask_ratio).float()
        elif mask_ratio == -1:
            mask_ratio_ = torch.rand(num_batch, device=self._device)[:, None]
            mask_ratio_ = 0.4 + mask_ratio_/2 # (0.4, 0.9)
            diffuse_mask = (u < mask_ratio_).float()
        else:
            diffuse_mask = torch.ones((num_batch, batch_length), device=self._device)
        return diffuse_mask

    def _decoupled_ot_SE3(self, trans_0, trans_1, rot_0, rot_1, res_mask):
        # a corrected version of batch_ot
        # decouple the num of noise the the number of samples

        # trans_0: noisy trans
        # trans_1: ground truth trans
        # rearrange the noisy trans and ground truth trans within a batch to get the optimal transport
        num_batch, num_res = trans_1.shape[:2]
        num_noise = trans_0.shape[0]

        batch_nm_0 = trans_0[:, None, :, :].expand(-1, num_batch, -1, -1).reshape(num_noise*num_batch, num_res, 3) # [num_noise, num_batch, 3] -> [num_noise*num_batch, 3]
        batch_nm_1 = trans_1[None, :, :, :].expand(num_noise, -1, -1, -1).reshape(num_noise*num_batch, num_res, 3) # [num_noise, num_batch, 3] -> [num_noise*num_batch, 3]
        batch_mask = res_mask[None, :, :].expand(num_noise, -1, -1).reshape(num_noise*num_batch, num_res) # [num_noise, num_batch] -> [num_noise*num_batch]
        aligned_nm_0, aligned_nm_1, _ = du.batch_align_structures(
            batch_nm_0, batch_nm_1, mask=batch_mask
        )
        aligned_nm_0 = aligned_nm_0.reshape(num_noise, num_batch, num_res, 3)
        aligned_nm_1 = aligned_nm_1.reshape(num_noise, num_batch, num_res, 3)
        batch_mask = batch_mask.reshape(num_noise, num_batch, num_res)

        trans_cost_matrix = torch.sum(
            torch.linalg.norm(aligned_nm_0 - aligned_nm_1, dim=-1), dim=-1
        ) / torch.sum(batch_mask, dim=-1) # [num_noise, num_batch], i means noise, j means ground truth
        noise_trans_index, gt_trans_index = linear_sum_assignment(du.to_numpy(trans_cost_matrix)) # gt_perm : 0 - num_batch-1; noise_perm : index * num_batch from [0 - num_noise-1]
        trans_pair_mapping = {g_i:n_i for n_i, g_i in zip(noise_trans_index, gt_trans_index)}
        optimal_trans = torch.stack([trans_0[trans_pair_mapping[g_i]] for g_i in range(num_batch)], dim=0)

        # get the optimal transport 
        # the noisy rotation and noisy translation are sampled independently, and thus the optimal transport could be calculated independently
        # considering the high computation time for OT on rot (double for loops), rot can be independently controlled to perform OT
        if rot_0 is None:
            optimal_rot = None
        else:
            num_rot_noise = rot_0.shape[0]
            rot_cost_matrix = torch.zeros(num_rot_noise, num_batch).to(self._device)
            for i in range(num_rot_noise):
                for j in range(num_batch):
                    so3_dist = torch.sum(
                        so3_relative_angle(rot_0[i], rot_1[j])
                    ) / res_mask[j].sum().item()
                    rot_cost_matrix[i,j] = so3_dist
            noise_rot_index, gt_rot_index = linear_sum_assignment(du.to_numpy(rot_cost_matrix))
            rot_pair_mapping = {g_i:n_i for n_i, g_i in zip(noise_rot_index, gt_rot_index)}
            optimal_rot = torch.stack([rot_0[rot_pair_mapping[g_i]] for g_i in range(num_batch)], dim=0)
        
        return optimal_trans, optimal_rot

    def _corrupt_SE3_with_decoupled_ot(self, trans_1, rotmats_1, t, res_mask, diffuse_mask, num_noise, do_rot_ot=False):
        # a corrected version of FoldFlow SE(3) corrupt with decoupled OT

        num_batch, num_res = res_mask.shape
        assert num_noise >= num_batch

        """ get trans_0 """
        sample_trans_nm_0 = _centered_gaussian(num_noise, num_res, self._device)
        sample_trans_0 = sample_trans_nm_0 * du.NM_TO_ANG_SCALE

        """ get rots_0 in FoldFlow style, the noise is sampled by scipy directly """
        sample_rotmats_0 = Rotation.random(num_noise*num_res).as_matrix()
        sample_rotmats_0 = torch.from_numpy(sample_rotmats_0).to(self._device, dtype=rotmats_1.dtype)
        sample_rotmats_0 = sample_rotmats_0.reshape(num_noise, num_res, 3, 3)

        if do_rot_ot:
            trans_0, rotmats_0 = self._decoupled_ot_SE3(sample_trans_0, trans_1, sample_rotmats_0, rotmats_1, res_mask)
        else:
            trans_0, _ = self._decoupled_ot_SE3(sample_trans_0, trans_1, None, rotmats_1, res_mask)
            rotmats_0 = sample_rotmats_0[:num_batch]

        """ corrupt trans """
        if self._trans_cfg.train_schedule == 'linear':
            trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        else:
            raise ValueError(
                f'Unknown trans schedule {self._trans_cfg.train_schedule}')
        trans_t = _trans_diffuse_mask(trans_t, trans_1, diffuse_mask)
        trans_t =  trans_t * res_mask[..., None]

        """ corrupt rots """
        so3_schedule = self._rots_cfg.train_schedule
        if so3_schedule == 'exp':
            so3_t = 1 - torch.exp(-t*self._rots_cfg.exp_rate)
        elif so3_schedule == 'linear':
            so3_t = t
        else:
            raise ValueError(f'Invalid schedule: {so3_schedule}')
        rotmats_t = so3_utils.geodesic_t(so3_t[..., None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=self._device)
        rotmats_t = (
            rotmats_t * res_mask[..., None, None]
            + identity[None, None] * (1 - res_mask[..., None, None])
        )
        rotmats_t = _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask)
    
        return trans_t, rotmats_t

    def _corrupt_aatypes(self, aatypes_1, t, res_mask, diffuse_mask):
        
        if not self._aatypes_cfg.corrupt:
            return aatypes_1

        num_batch, num_res = res_mask.shape
        assert aatypes_1.shape == (num_batch, num_res)
        assert t.shape == (num_batch, 1)
        # assert res_mask.shape == (num_batch, num_res)
        # assert diffuse_mask.shape == (num_batch, num_res)

        if self._aatypes_cfg.interpolant_type == "masking":
            u = torch.rand(num_batch, num_res, device=self._device)
            aatypes_t = aatypes_1.clone()
            corruption_mask = u < (1 - t) # (B, N)

            aatypes_t[corruption_mask] = du.MASK_TOKEN_INDEX

            aatypes_t = aatypes_t * res_mask + du.MASK_TOKEN_INDEX * (1 - res_mask)

        elif self._aatypes_cfg.interpolant_type == "uniform":
            u = torch.rand(num_batch, num_res, device=self._device)
            aatypes_t = aatypes_1.clone()
            corruption_mask = u < (1-t) # (B, N)
            mask_tk_mask = aatypes_t.eq(du.NUM_TOKENS)
            uniform_sample = torch.randint_like(aatypes_t, low=0, high=du.NUM_TOKENS)
            aatypes_t[corruption_mask] = uniform_sample[corruption_mask]
            aatypes_t[mask_tk_mask] = uniform_sample[mask_tk_mask]

            # aatypes_t = aatypes_t * res_mask + du.MASK_TOKEN_INDEX * (1 - res_mask)
        else:
            raise ValueError(f"Unknown aatypes interpolant type {self._aatypes_cfg.interpolant_type}")

        return _aatypes_diffuse_mask(aatypes_t, aatypes_1, diffuse_mask)
        # return _aatypes_diffuse_mask(aatypes_t, aatypes_1, torch.ones_like(res_mask))

    def _corrupt_se3(self, trans_1, r3_t, rotmats_1, so3_t, res_mask, diffuse_mask, rot_style):
        
        num_batch, num_res, _ = trans_1.shape
        # num_noise = int(1.2 * num_batch)

        if rot_style == 'multiflow':

            # sample trans_0
            trans_nm_0 = _centered_gaussian(num_batch, num_res, self._device)
            trans_0 = trans_nm_0 * du.NM_TO_ANG_SCALE

            # sample rotmats_0
            # noisy_rotmats = self.igso3.sample(torch.tensor([1.5]), num_batch*num_res).to(self._device)
            # noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)
            # rotmats_0 = torch.einsum(
            #     "...ij,...jk->...ik", rotmats_1, noisy_rotmats) # [B, L, 3, 3]
            rotmats_0 = self.sample_rand_rot(num_batch, num_res, rotmats_1.dtype)
            
            # decoupled batch ot
            if self._rots_cfg.batch_ot:
                trans_0, rotmats_0 = self._decoupled_ot_SE3(trans_0, trans_1, rotmats_0, rotmats_1, res_mask)
            else:
                trans_0, _ = self._decoupled_ot_SE3(trans_0, trans_1, None, rotmats_1, res_mask)
            # corrupt trans
            if self._trans_cfg.train_schedule == 'linear':
                trans_t = (1 - r3_t[..., None]) * trans_0 + r3_t[..., None] * trans_1
            else:
                raise ValueError(
                    f'Unknown trans schedule {self._trans_cfg.train_schedule}')
            trans_t = _trans_diffuse_mask(trans_t, trans_1, diffuse_mask)
            trans_t = trans_t * res_mask[..., None]

            # corrupt rotmats
            so3_schedule = self._rots_cfg.train_schedule
            if so3_schedule == 'exp':
                so3_t = 1 - torch.exp(-so3_t*self._rots_cfg.exp_rate)
            elif so3_schedule == 'linear':
                so3_t = so3_t
            else:
                raise ValueError(f'Invalid schedule: {so3_schedule}')
            rotmats_t = so3_utils.geodesic_t(so3_t[..., None], rotmats_1, rotmats_0)
            identity = torch.eye(3, device=self._device)
            rotmats_t = (
                rotmats_t * res_mask[..., None, None]
                + identity[None, None] * (1 - res_mask[..., None, None])
            )
            rotmats_t = _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask)
            if torch.any(torch.isnan(rotmats_t)):
                raise ValueError('NaN in rotmats_t during corruption')

            if not self._trans_cfg.corrupt:
                trans_t = trans_1
            
            if not self._rots_cfg.corrupt:
                rotmats_t = rotmats_1
        
        elif rot_style == 'foldflow':
            # trans_t, rotmats_t = self._corrupt_SE3(trans_1, rotmats_1, so3_t, res_mask, diffuse_mask)
            num_batch = trans_1.shape[0]
            # num_noise = max(2*num_batch, 64)
            trans_t, rotmats_t = self._corrupt_SE3_with_decoupled_ot(trans_1, rotmats_1, so3_t, res_mask, diffuse_mask, num_noise, do_rot_ot=self._rots_cfg.batch_ot)
            if torch.any(torch.isnan(trans_t)):
                raise ValueError('NaN in trans_t during corruption')
            if torch.any(torch.isnan(rotmats_t)):
                raise ValueError('NaN in rotmats_t during corruption')
        
        else:
            raise ValueError(f'{rot_style} has not been implemented')

        return trans_t, rotmats_t

    def _corrupt_torsions(self, torsions_1, t, res_mask, diffuse_mask):
        num_batch, num_res = res_mask.shape
        assert torsions_1.shape == (num_batch, num_res, 4)
        assert t.shape == (num_batch, 1)
        assert res_mask.shape == (num_batch, num_res)
        assert diffuse_mask.shape == (num_batch, num_res)
            
        # u = torch.rand(num_batch, num_res, device=self._device)
        # torsions_t = torsions_1.clone()
        # corruption_mask = u < (1-t) # (B, N)
        # TODO: add corruption_mask as in aatype

        # the input t is sample from [sidechain_start_t, 1], is need to be transformed to [0, 1] when corrupt torsion
        corrupt_t = (t - self.start_t['sidechain']) / (1 - self.start_t['sidechain'])
        corrupt_t = torch.clamp(corrupt_t, min=self._cfg.min_t, max=self.end_t)
        
        torsions_0 = torch.rand_like(torsions_1) * 2 * np.pi
        torsions_0 = so2_utils.mod_to_standard_angle_range(torsions_0)
        
        torsions_t = so2_utils.interpolate(a=torsions_0, b=torsions_1, t=corrupt_t[...,None])

        torsions_t = torsions_t * res_mask[...,None] + torsions_0 * (1 - res_mask[...,None])

        return _torsions_diffuse_mask(torsions_t, torsions_1, diffuse_mask) 

    def modify_t(self, t, dt, min_t):
        # t : [B, 1]
        # dt : float
        # min_t : float
        # continuously add dt to t until t >= min_t
        cal_t = t + dt * torch.ceil((min_t - t) / dt).clamp(0)
        return cal_t

    def corrupt_batch(self, batch, rot_style='multiflow', training_models=('backbone', ), forward_dt=0.002, forward_steps=0):
        curr_support_tasks = []
        for training_model in training_models:
            assert training_model in ('backbone', 'sidechain', 'refine')
        training_model = '-'.join(sorted(training_models, key=lambda x: {'backbone': 0, 'sidechain': 1, 'refine': 2}[x]))
        curr_support_tasks = self.model_support_tasks[training_model]
        curr_task_prob = self.norm_task_prob(curr_support_tasks)
        curr_data_modalities = self.data_modality[training_model]
        merge_data_modalities = sum(curr_data_modalities, [])
        training_model_start_t = min([self.start_t[tm] for tm in training_models])

        noisy_batch = copy.deepcopy(batch)

        # Groud truth data
        # [B, N, 3]
        trans_1 = batch['trans_1']  # Angstrom
        # [B, N, 3, 3]
        rotmats_1 = batch['rotmats_1']
        # [B, N]
        aatypes_1 = batch['aatypes_1']
        # [B, N, 4]
        torsions_1 = batch['torsions_1']


        # [B, N]
        res_mask = batch['res_mask']
        res_mask_aa = batch['res_mask_aa']
        diffuse_mask = batch['diffuse_mask']
        diffuse_mask_aa = batch['diffuse_mask_aa']
        num_batch, num_res = diffuse_mask.shape

        # Sample t
        # [B, 1]
        t_lib = {}
        if self._cfg.codesign_separate_t:
            # if separate_t, different modality could be assigned with different t for special tasks, but not all the t are different
            ### in backbone model traning: rot_t and trans_t are the same, cat_t is different
            ### in sidechain model traning: rot_t and tor_t and cat_t are the same, tor_t is different, and tor_t should lower than other t as sidechain is dependent on backbone
            ### in refine model traning: t is same to backbone model

            # build mask for special tasks and different model support different tasks
            # tasks in backbone model training: codesign, folding, inverse-folding
            # tasks in sidechain model training: codesign (full-atom), packing
            # tasks in refine model training: codesign (full-atom), folding (full-atom), inverse-folding
            u = torch.rand((num_batch,), device=self._device)
            cumulative_prob = 0.0
            task_mask = {}
            for task_ in curr_task_prob:
                task_mask[task_] = (u < cumulative_prob + curr_task_prob[task_]).float() * \
                (u >= cumulative_prob).float() # [B,]
                cumulative_prob += curr_task_prob[task_]
            if 'codesign' in task_mask:
                codesign_mask = 1
                for task_ in task_mask:
                    if task_ != 'codesign':
                        codesign_mask = codesign_mask - task_mask[task_]
                task_mask['codesign'] = codesign_mask
            
            # build t for different data modalities in different tasks
            for data_modality in curr_data_modalities:
                # sample t for each modality separately
                start_t = training_model_start_t
                modality_t = self.sample_t_with_range(num_batch, start_t, self.end_t) # [B, ]
                modality_t = modality_t - forward_dt * forward_steps
                for iso_modality in data_modality:
                    if iso_modality == 'torsions': 
                        continue # torsions need to be processed individually
                    t_lib[self.t_mapping[iso_modality]] = copy.deepcopy(modality_t) # [B, ]
            
            # add t for torsions
            mininum_t_mask = t_lib['cat_t'].lt(t_lib['r3_t']).float()
            t_lib['tor_t'] = t_lib['cat_t'].clone() * mininum_t_mask + t_lib['r3_t'].clone() * (1-mininum_t_mask)

            ### folding task: 
            if 'folding' in task_mask:
                t_lib['cat_t'] = torch.ones_like(t_lib['cat_t']) * task_mask['folding'] + t_lib['cat_t'] * (1 - task_mask['folding']) # cat_t is set to 1 for folding task
                t_lib['tor_t'] = t_lib['tor_t'] * (1 - task_mask['folding']) + t_lib['r3_t'] * task_mask['folding'] # tor_t is set to r3_t for folding task

            ### inverse-folding task: 
            if 'inverse-folding' in task_mask:
                t_lib['r3_t'] = torch.ones_like(t_lib['r3_t']) * task_mask['inverse-folding'] + t_lib['r3_t'] * (1 - task_mask['inverse-folding']) # r3_t is set to 1 for inverse-folding task
                t_lib['so3_t'] = torch.ones_like(t_lib['so3_t']) * task_mask['inverse-folding'] + t_lib['so3_t'] * (1 - task_mask['inverse-folding']) # so3_t is set to 1 for inverse-folding task
                t_lib['tor_t'] = t_lib['tor_t'] * (1 - task_mask['inverse-folding']) + t_lib['cat_t'] * task_mask['inverse-folding'] # tor_t is set to cat_t for inverse-folding task

            ### packing task: 
            if 'packing' in task_mask:
                t_lib['cat_t'] = torch.ones_like(t_lib['cat_t']) * task_mask['packing'] + t_lib['cat_t'] * (1 - task_mask['packing']) # cat_t is set to 1 for packing task
                t_lib['r3_t'] = torch.ones_like(t_lib['r3_t']) * task_mask['packing'] + t_lib['r3_t'] * (1 - task_mask['packing']) # r3_t is set to 1 for packing task
                t_lib['so3_t'] = torch.ones_like(t_lib['so3_t']) * task_mask['packing'] + t_lib['so3_t'] * (1 - task_mask['packing']) # so3_t is set to 1 for packing task

        else:
            # if not separate_t, all data modalities share the same t within each sample
            unified_t = self.sample_t_with_range(num_batch, training_model_start_t, self.end_t)
            unified_t = unified_t - forward_dt * forward_steps
            for iso_modality in merge_data_modalities:
                t_lib[self.t_mapping[iso_modality]] = unified_t.clone()
        
        # we always provide the tor_t in any situation
        # this design is intended to support inference for all models during the roll out when training the backbone model
        # in sidechain model training, tor_t is sampled, tor_t is used for training in this case
        # in bacbone model training, tor_t is depends on the minimum of cat_t and r3_t/so3_t, tor_t is only used for inference in rollout
        if 'tor_t' not in t_lib: # only happens in backbone model
            min_t_mask = t_lib['cat_t'].lt(t_lib['r3_t']).float()
            t_lib['tor_t'] = min_t_mask * t_lib['cat_t'] + (1 - min_t_mask) * t_lib['r3_t']
        
        for t_ in t_lib:
            noisy_batch[t_] = t_lib[t_][:, None] # [B, 1]
            
        # Corrupt data
        # the data is actually corrupted with the mod_t, but will not record mod_t
        for iso_modality in ['aatypes', 'trans', 'rotmats', 'torsions']:
            
            if 'refine' in training_models:
                bb_start_t = self.start_t['refine']
            else:
                bb_start_t = self.start_t['backbone']

            if iso_modality == 'aatypes':
                cat_t = self.modify_t(noisy_batch['cat_t'], forward_dt, bb_start_t)
                noisy_batch['aatypes_t'] = self._corrupt_aatypes(
                    aatypes_1, cat_t, res_mask_aa, diffuse_mask_aa)
            elif iso_modality == 'trans':
                strcut_t = self.modify_t(noisy_batch['r3_t'], forward_dt, bb_start_t)
                trans_t, rotmats_t = self._corrupt_se3(trans_1=trans_1, 
                                                        r3_t=strcut_t, 
                                                        rotmats_1=rotmats_1, 
                                                        so3_t=strcut_t, 
                                                        res_mask=res_mask, 
                                                        diffuse_mask=diffuse_mask, 
                                                        rot_style=rot_style)
                noisy_batch['trans_t'] = trans_t
                noisy_batch['rotmats_t'] = rotmats_t
            elif iso_modality == 'torsions':
                tor_t = self.modify_t(noisy_batch['tor_t'], forward_dt, self.start_t['sidechain'])
                noisy_batch['torsions_t'] = self._corrupt_torsions(
                    torsions_1, tor_t, res_mask, diffuse_mask)
            # else:
            #     noisy_batch[iso_modality+'_t'] = None

        # Build self-condition
        noisy_batch['aatypes_sc'] = torch.zeros_like(
            aatypes_1)[..., None].repeat(1, 1, self.num_tokens)
        noisy_batch['trans_sc'] = torch.zeros_like(trans_1)
        noisy_batch['rotvecs_sc'] = torch.zeros_like(trans_1)
        noisy_batch['torsions_sc'] = torch.zeros_like(torsions_1)
        return noisy_batch

    def corrupt_aatypes_for_adj_t(self, aatypes_1, cat_t, res_mask_aa, diffuse_mask_aa):
        aatypes_adj_t = self._corrupt_aatypes(aatypes_1, cat_t, res_mask_aa, diffuse_mask_aa)
        aatypes_sc_adj_t = torch.zeros_like(
            aatypes_1)[..., None].repeat(1, 1, self.num_tokens)
        return aatypes_adj_t, aatypes_sc_adj_t

    def corrupt_se3_for_adj_t(self, trans_1, r3_t, rotmats_1, so3_t, torsions_1, res_mask, diffuse_mask, rot_style='multiflow'):
        noisy_se3_adj_t = {}
        trans_t, rotmats_t = self._corrupt_se3(trans_1=trans_1, 
                                                r3_t=r3_t, 
                                                rotmats_1=rotmats_1, 
                                                so3_t=so3_t, 
                                                res_mask=res_mask, 
                                                diffuse_mask=diffuse_mask, 
                                                rot_style=rot_style)
        noisy_se3_adj_t['trans_t'] = trans_t
        noisy_se3_adj_t['rotmats_t'] = rotmats_t
        noisy_se3_adj_t['torsions_t'] = torch.zeros_like(torsions_1)

        noisy_se3_adj_t['trans_sc'] = torch.zeros_like(trans_1)
        noisy_se3_adj_t['rotvecs_sc'] = torch.zeros_like(trans_1)
        noisy_se3_adj_t['torsions_sc'] = noisy_se3_adj_t['torsions_t']
        
        return noisy_se3_adj_t

    def corrupt_batch_with_adj_t(self, org_noisy_batch, delta_t, rot_style='multiflow'):
        
        adj_t_noisy_batch = {}
        trans_1 = org_noisy_batch['trans_1']
        rotmats_1 = org_noisy_batch['rotmats_1']
        torsions_1 = org_noisy_batch['torsions_1']

        # adj_cat_t = org_noisy_batch['cat_t'].clone() + delta_t
        # adj_cat_t = adj_cat_t.clamp(min=self.start_t['backbone'], max=1)
        adj_r3_t = org_noisy_batch['r3_t'].clone() + delta_t
        adj_cat_t = adj_cat_t.clamp(min=self.start_t['backbone'], max=1)
        adj_so3_t = org_noisy_batch['so3_t'].clone() + delta_t
        adj_so3_t = adj_so3_t.clamp(min=self.start_t['backbone'], max=1)
        adj_tor_t = org_noisy_batch['tor_t'].clone() + delta_t
        adj_tor_t = adj_tor_t.clamp(min=self.start_t['sidechain'], max=1)

        # adj_t_noisy_batch['aatypes_t'] = self._corrupt_aatypes(
        #     adj_t_noisy_batch['aatypes_1'], adj_cat_t, adj_t_noisy_batch['res_mask'], adj_t_noisy_batch['diffuse_mask'])
        # adj_t_noisy_batch['cat_t'] = adj_cat_t
        
        trans_t, rotmats_t = self._corrupt_se3(trans_1=trans_1,
                                                r3_t=adj_r3_t,
                                                rotmats_1=rotmats_1,
                                                so3_t=adj_so3_t,
                                                res_mask=org_noisy_batch['res_mask'],
                                                diffuse_mask=org_noisy_batch['diffuse_mask'], 
                                                rot_style=rot_style)
        adj_t_noisy_batch['trans_t'] = trans_t
        adj_t_noisy_batch['r3_t'] = adj_r3_t
        adj_t_noisy_batch['rotmats_t'] = rotmats_t
        adj_t_noisy_batch['so3_t'] = adj_so3_t

        # adj_t_noisy_batch['torsions_t'] = self._corrupt_torsions(
        #     adj_t_noisy_batch['torsions_1'], adj_tor_t, adj_t_noisy_batch['res_mask'], adj_t_noisy_batch['diffuse_mask'])
        adj_t_noisy_batch['torsions_t'] = org_noisy_batch['torsions_t']
        adj_t_noisy_batch['tor_t'] = adj_tor_t

        adj_t_noisy_batch['trans_sc'] = torch.zeros_like(trans_1)
        adj_t_noisy_batch['rotvecs_sc'] = torch.zeros_like(trans_1)
        adj_t_noisy_batch['torsions_sc'] = torch.zeros_like(torsions_1)
        
        return adj_t_noisy_batch

    def rot_sample_kappa(self, t):
        if self._rots_cfg.sample_schedule == 'exp':
            return 1 - torch.exp(-t*self._rots_cfg.exp_rate)
        elif self._rots_cfg.sample_schedule == 'linear':
            return t
        else:
            raise ValueError(
                f'Invalid schedule: {self._rots_cfg.sample_schedule}')

    def _trans_vector_field(self, t, trans_1, trans_t):
        if self._trans_cfg.sample_schedule == 'linear':
            # in rollout mode, it may happens that a modality with t=1 needs to take a euler step
            # in original code, the euler step only occurs in sampling process
            #       if a modality has the t of 1, it means that we are excuting a special task, like folding
            #       in this case, the modality will be always replaced by the ground truth, the euler step is not needed
            # in rollout mode, which means we are training the model
            #       the t within one batch is different, which means some sample is used for folding training, some for co-design training
            #       in this case, the euler step will be applied to all modalities of all samples
            #       (unless we use a for loop to iterate process each individual sample, but it is not recommended)
            #       so we must solve the problem of t=1 in calculating the vector field in euler step
            #       the solution is add a eps to 1-t to avoid dividing by zero
            #       this modification will only affect t=1
            #       then the sample with t=1 will be special processed in rollout by keeping the ground truth

            t_mask = t.lt(1).float()
            t_ = t - self._trans_cfg.eps * (1 - t_mask)
            trans_vf = (trans_1 - trans_t) / (1 - t_)
            trans_vf = trans_vf * t_mask + torch.zeros_like(trans_vf) * (1 - t_mask)
        elif self._trans_cfg.sample_schedule == 'vpsde':
            bmin = self._trans_cfg.vpsde_bmin
            bmax = self._trans_cfg.vpsde_bmax
            bt = bmin + (bmax - bmin) * (1-t) # scalar
            alpha_t = torch.exp(- bmin * (1-t) - 0.5 * (1-t)**2 * (bmax - bmin)) # scalar
            trans_vf = 0.5 * bt * trans_t + \
                0.5 * bt * (torch.sqrt(alpha_t) * trans_1 - trans_t) / (1 - alpha_t)
        else:
            raise ValueError(
                f'Invalid sample schedule: {self._trans_cfg.sample_schedule}'
            )
        return trans_vf

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        assert d_t >= 0
        trans_vf = self._trans_vector_field(t, trans_1, trans_t)
        return trans_t + trans_vf * d_t
    
    def _torsions_euler_step(self, d_t, t, torsions_1, torsions_t):
        # TODO: verify the t and d_t in torsion euler process,
        #       as the torsion are actually noised with t scaled to [min_t, max_t],
        #       then the t and d_t should also be scaled accordingly in the reverse process ?
        corrupt_t = (t - self.start_t['sidechain']) / (1 - self.start_t['sidechain'])
        corrupt_t = torch.clamp(corrupt_t, min=self._cfg.min_t, max=self.end_t)

        corrupt_d_t = d_t / (1 - self.start_t['sidechain'])
        
        assert d_t >= 0
        # TODO (zhouxiangxin): handle period of vector field
        torsions_vf = so2_utils.calc_torus_vf(x1=torsions_1, xt=torsions_t, t=corrupt_t)
        torsions_next = torsions_t + torsions_vf * corrupt_d_t
        torsions_next = so2_utils.mod_to_standard_angle_range(torsions_next)
        
        return torsions_next

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):

        t_mask = t.lt(1).float()
        t_ = t - self._rots_cfg.eps * (1 - t_mask)

        if self._rots_cfg.sample_schedule == 'linear':
            scaling = 1 / (1 - t_)
        elif self._rots_cfg.sample_schedule == 'exp':
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(
                f'Unknown sample schedule {self._rots_cfg.sample_schedule}')
        # TODO: Add in SDE.
        return so3_utils.geodesic_t(
            scaling * d_t, rotmats_1, rotmats_t)

    def _regularize_step_probs(self, step_probs, aatypes_t):
        batch_size, num_res, S = step_probs.shape
        device = step_probs.device
        assert aatypes_t.shape == (batch_size, num_res)

        step_probs = torch.clamp(step_probs, min=0.0, max=1.0) # [B, L, S]
        # TODO replace with torch._scatter
        step_probs[
            torch.arange(batch_size, device=device).repeat_interleave(num_res), # [B*L]
            torch.arange(num_res, device=device).repeat(batch_size), # [L]
            aatypes_t.long().flatten() # [B*L]
        ] = 0.0
        step_probs[
            torch.arange(batch_size, device=device).repeat_interleave(num_res),
            torch.arange(num_res, device=device).repeat(batch_size),
            aatypes_t.long().flatten()
        ] = 1.0 - torch.sum(step_probs, dim=-1).flatten()
        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
        return step_probs

    def _aatypes_euler_step_APM(self, d_t, t, logits_1, noise_scale=1.0, temperature=1.0, argmax=False, diffuse_mask=None, apply_penalty=True):
        batch_size, num_res, S = logits_1.shape
        data_shape = logits_1.shape[:2]
        assert S == 21
        assert self._aatypes_cfg.interpolant_type == "masking"
        device = logits_1.device

        if not diffuse_mask is None:
            num_res = int(diffuse_mask[0].sum().cpu().item())

        number_total_unmask = (d_t + t) * num_res
        number_total_unmask = torch.ceil(number_total_unmask).int().item()

        logits_1_wo_mask = logits_1[:, :, 0:-1] # (B, D, S-1)

        if apply_penalty:
            self.reference_normalized_counts = self.reference_normalized_counts.to(device)
            penalty_factor = self._aatypes_cfg.penalty_factor
            penalty_mode = self._aatypes_cfg.penalty_mode

            logits_1_wo_mask = F.log_softmax(logits_1_wo_mask, dim=-1)

            # current token predictions
            if not argmax:
                dist_temp = torch.distributions.Categorical(logits=logits_1_wo_mask.div(temperature))
                cur_tokens = dist_temp.sample()  # [B, D]
            else:
                cur_tokens = torch.argmax(logits_1_wo_mask, dim=-1)  # [B, D]

            # initialize penalty
            freq_penalty = torch.zeros_like(logits_1_wo_mask)  # [B, D, S-1]
            
            # compute penalty
            for b in range(batch_size):
                # calculate frequency of each token
                token_counts = torch.bincount(cur_tokens[b], minlength=S-1)  # [S-1]
                token_freq = token_counts / token_counts.sum()  # [S-1]
                
                # smooth frequency, avoiding division by zero
                epsilon = 1e-6
                token_freq_smooth = token_freq + epsilon
                reference_smooth = self.reference_normalized_counts + epsilon
                
                # calculate KL divergence style penalty
                kl_term = token_freq_smooth * torch.log(token_freq_smooth / reference_smooth)  # [S-1]
                
                # apply penalty with mode
                if penalty_mode == 'exceed_only':
                    exceed_mask = token_freq > self.reference_normalized_counts  # [S-1]
                    penalty = penalty_factor * kl_term * exceed_mask.float()
                elif penalty_mode == 'both':
                    penalty = penalty_factor * kl_term
                else:
                    raise ValueError("Invalid penalty_mode. Choose 'exceed_only' or 'both'.")
                
                freq_penalty[b] = penalty[None, :]  # [1, S-1] -> [D, S-1]
            
            # adjust logits with penalty
            logits_1_wo_mask = logits_1_wo_mask - freq_penalty  # [B, D, S-1]

        if argmax:
            all_scores = torch.log(F.softmax(logits_1_wo_mask, dim=-1)) # (B, D, S-1)
            scores, tokens = torch.max(all_scores, dim=-1) # (B, D)
            scores_org = scores
        else:
            dist_temp = torch.distributions.Categorical(logits=logits_1_wo_mask.div(temperature)) # temperature scaled logits for tokens sampling
            tokens = dist_temp.sample() # [B, D]
            scores = dist_temp.log_prob(tokens) # [B, D]
            dist_org = torch.distributions.Categorical(logits=logits_1_wo_mask) # original logits for score comparsion
            scores_org = dist_org.log_prob(tokens) # [B, D]

        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        scores += noise_scale * gumbel_noise

        if not diffuse_mask is None:
            scores = scores * diffuse_mask + -1e8 * (1 - diffuse_mask) 

        _, unmask_indices = torch.topk(scores, k=number_total_unmask, dim=-1) # unmask_indices: [B, N_unmask]

        sample_base = torch.arange(batch_size)[:,None] * data_shape[1]
        sample_base = sample_base.repeat(1, number_total_unmask).view(-1).to(device) # [B, N_unmask] -> [B*N_unmask]
        unmask_indices_flatten = unmask_indices.view(-1) + sample_base # [B, N_unmask] -> [B*N_unmask]
        unified_mask = torch.zeros((batch_size, logits_1.shape[1]), device=device).view(-1)
        unified_mask[unmask_indices_flatten.long()] = 1
        unified_mask = unified_mask.view(*data_shape) # [B, D]

        aatypes_t = tokens * unified_mask + du.MASK_TOKEN_INDEX * (1 - unified_mask)

        unmask_scores = scores_org * unified_mask
        unmask_scores = unmask_scores.sum(-1)

        return aatypes_t, unmask_scores

    def _aatypes_euler_step_APM_Refine(self, d_t, t, backbone_logits_1, refine_logits_1, noise_scale=1.0, temperature=1.0, argmax=False, diffuse_mask=None):
        batch_size, num_res, S = backbone_logits_1.shape
        data_shape = backbone_logits_1.shape[:2]
        assert S == 21
        assert self._aatypes_cfg.interpolant_type == "masking"
        device = backbone_logits_1.device

        if not diffuse_mask is None:
            num_res = int(diffuse_mask[0].sum().cpu().item())

        number_total_unmask = (d_t + t) * num_res
        number_total_unmask = torch.ceil(number_total_unmask).int().item()
        number_total_unmask = min(number_total_unmask, num_res)
        number_total_scaled = num_res - number_total_unmask

        logits_1_wo_mask = backbone_logits_1[:, :, 0:-1] # (B, D, S-1)
        refine_logits_1_wo_mask = refine_logits_1[:, :, 0:-1] # (B, D, S-1)

        # number_total_scaled = 0
        # if number_total_scaled > 0:

        #     delta_logits_1_wo_mask = refine_logits_1_wo_mask - logits_1_wo_mask
        #     delta_logits_1_module_wo_mask = torch.sqrt(torch.sum(delta_logits_1_wo_mask ** 2, dim=-1)) # (B, D)
        #     # delta_logits_tempature_scale = F.tanh(delta_logits_1_wo_mask) * 2 # ~(0,1)
        #     scale_base_factor, _ = delta_logits_1_module_wo_mask.topk(number_total_scaled, dim=-1) # (B, D*0.2)
        #     scale_base_factor, _ = scale_base_factor.min(dim=-1) # (B)
        #     scale_factor = delta_logits_1_module_wo_mask / scale_base_factor[..., None] # (B)
        #     scale_factor = torch.clamp(scale_factor, min=1)

        #     logits_1_wo_mask = logits_1_wo_mask / scale_factor[..., None]

        logits_1_wo_mask = logits_1_wo_mask*0.8 + refine_logits_1_wo_mask*0.2

        if argmax:
            all_scores = torch.log(F.softmax(logits_1_wo_mask, dim=-1)) # (B, D, S-1)
            scores, tokens = torch.max(all_scores, dim=-1) # (B, D)
            scores_org = scores
        else:

            # logits_merge = 0.8*logits_1_wo_mask + 0.2*refine_logits_1_wo_mask
            logits_merge = logits_1_wo_mask

            dist_temp = torch.distributions.Categorical(logits=logits_merge.div(temperature)) # temperature scaled logits for tokens sampling
            # dist_temp_refine = torch.distributions.Categorical(logits=refine_logits_1_wo_mask.div(temperature)) # temperature scaled logits for tokens sampling

            dist_org = torch.distributions.Categorical(logits=logits_1_wo_mask) # original logits for score comparsion
            # dist_org_refine = torch.distributions.Categorical(logits=refine_logits_1_wo_mask)

            tokens = dist_temp.sample() # [B, D]
            scores = dist_temp.log_prob(tokens) # [B, D]
            # refine_scores = dist_temp_refine.log_prob(tokens) # [B, D]

            # org_scores = dist_org.log_prob(tokens) # [B, D]
            # org_refine_scores = dist_org_refine.log_prob(tokens) # [B, D]
            # delta_scores = torch.abs(scores - refine_scores)


            # refined_scores = dist_temp_refine.log_prob(tokens) # [B, D]
            # scores = 0.8 * scores + 0.2 * refined_scores
            # scores = scores - delta_scores * (1-t)
            scores_org = dist_org.log_prob(tokens) # [B, D]

        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        scores += noise_scale * gumbel_noise

        if not diffuse_mask is None:
            scores = scores * diffuse_mask + -1e8 * (1 - diffuse_mask) 

        _, unmask_indices = torch.topk(scores, k=number_total_unmask, dim=-1) # unmask_indices: [B, N_unmask]

        sample_base = torch.arange(batch_size)[:,None] * data_shape[1]
        sample_base = sample_base.repeat(1, number_total_unmask).view(-1).to(device) # [B, N_unmask] -> [B*N_unmask]
        unmask_indices_flatten = unmask_indices.view(-1) + sample_base # [B, N_unmask] -> [B*N_unmask]
        unified_mask = torch.zeros(data_shape, device=device).view(-1)
        unified_mask[unmask_indices_flatten.long()] = 1
        unified_mask = unified_mask.view(*data_shape) # [B, D]

        aatypes_t = tokens * unified_mask + du.MASK_TOKEN_INDEX * (1 - unified_mask)

        unmask_scores = scores_org * unified_mask
        unmask_scores = unmask_scores.sum(-1)

        return aatypes_t, unmask_scores

    def _aatypes_euler_step_APM_Refine_guide(self, d_t, t, backbone_logits_1, refine_logits_1, noise_scale=1.0, temperature=1.0, argmax=False):
        batch_size, num_res, S = backbone_logits_1.shape
        assert S == 21
        assert self._aatypes_cfg.interpolant_type == "masking"
        device = backbone_logits_1.device

        number_total_unmask = (d_t + t) * num_res
        number_total_unmask = torch.ceil(number_total_unmask).int().item()
        number_total_scaled = int((num_res - number_total_unmask)/2)

        logits_1_wo_mask = backbone_logits_1[:, :, 0:-1] # (B, D, S-1)
        refine_logits_1_wo_mask = refine_logits_1[:, :, 0:-1] # (B, D, S-1)

        if number_total_scaled > 0:

            delta_logits_1_wo_mask = refine_logits_1_wo_mask - logits_1_wo_mask
            delta_logits_1_module_wo_mask = torch.sqrt(torch.sum(delta_logits_1_wo_mask ** 2, dim=-1)) # (B, D)
            scale_base_factor, _ = delta_logits_1_module_wo_mask.topk(number_total_scaled, dim=-1) # (B, D*0.2)
            scale_base_factor, _ = scale_base_factor.min(dim=-1) # (B)
            scale_indicator = delta_logits_1_module_wo_mask.gt(scale_base_factor[:, None]).long() # (B, D)

        logits_merge = logits_1_wo_mask*0.8 + refine_logits_1_wo_mask*0.2

        # argmax source
        argmax_scores = torch.log(F.softmax(logits_merge, dim=-1)) # (B, D, S-1)
        argmax_score, argmax_tokens = torch.max(argmax_scores, dim=-1) # (B, D)

        # tempatrue source
        if number_total_scaled > 0:
            logits_merge = logits_1_wo_mask

            dist_temp = torch.distributions.Categorical(logits=logits_merge.div(temperature)) # temperature scaled logits for tokens sampling

            sampled_tokens = dist_temp.sample() # [B, D]
            sampled_score = dist_temp.log_prob(sampled_tokens) # [B, D]

            # merge two source
            merge_score = argmax_score * scale_indicator.float() + sampled_score * (1-scale_indicator.float())
            merge_token = argmax_tokens * scale_indicator + sampled_tokens * (1-scale_indicator)
        else:
            merge_score = argmax_score
            merge_token = argmax_tokens


        gumbel_noise = -torch.log(-torch.log(torch.rand_like(merge_score) + 1e-8) + 1e-8)
        merge_score += noise_scale * gumbel_noise
        _, unmask_indices = torch.topk(merge_score, k=number_total_unmask, dim=-1) # unmask_indices: [B, N_unmask]

        sample_base = torch.arange(batch_size)[:,None] * num_res
        sample_base = sample_base.repeat(1, number_total_unmask).view(-1).to(device) # [B, N_unmask] -> [B*N_unmask]
        unmask_indices_flatten = unmask_indices.view(-1) + sample_base # [B, N_unmask] -> [B*N_unmask]
        unified_mask = torch.zeros((batch_size, num_res), device=device).view(-1)
        unified_mask[unmask_indices_flatten.long()] = 1
        unified_mask = unified_mask.view(batch_size, num_res) # [B, D]

        aatypes_t = merge_token * unified_mask + du.MASK_TOKEN_INDEX * (1 - unified_mask)

        return aatypes_t, None
    
    def sample_backbone(
            self,
            num_batch,
            num_res,
            model,
            model_order, 
            num_timesteps=None,
            trans_0=None,
            rotmats_0=None,
            aatypes_0=None,
            torsions_0=None,
            trans_1=None,
            rotmats_1=None,
            aatypes_1=None,
            torsions_1=None,
            diffuse_mask=None,
            chain_idx=None,
            res_idx=None,
            t_nn=None,
            forward_folding=False,
            inverse_folding=False,
            packing=False,
            separate_t=False,
            rot_style='multiflow',
            PLM_embedder=None,
            PLM_type=None, 
            PLM_encoding=None, 
            task='unconditional', 
            PLM_templates=None
        ):

        tor_pred = False
        
        res_mask = torch.ones(num_batch, num_res, device=self._device)

        # Set-up initial prior samples
        if trans_0 is None:
            trans_0 = _centered_gaussian(
                num_batch, num_res, self._device) * du.NM_TO_ANG_SCALE
        if rotmats_0 is None:
            # rotmats_0 = _uniform_so3(num_batch, num_res, self._device)
            rotmats_0 = self.sample_rand_rot(num_batch, num_res, dtype=torch.float)
        if torsions_0 is None:
            torsions_0 = _uniform_torsion(num_batch, num_res, self._device)
        if aatypes_0 is None:
            # according to the previous results, we only use masking noise for aatypes
            aatypes_0 = _masked_categorical(num_batch, num_res, self._device)
        if res_idx is None:
            res_idx = torch.arange(
                num_res,
                device=self._device,
                dtype=torch.float32)[None].repeat(num_batch, 1)
            res_idx += 1 # index in traning processing is 1-based
        
        if chain_idx is None:
            chain_idx = res_mask

        if diffuse_mask is None:
            diffuse_mask = res_mask

        trans_sc = torch.zeros(num_batch, num_res, 3, device=self._device)
        aatypes_sc = torch.zeros(
            num_batch, num_res, self.num_tokens, device=self._device)
        rotvecs_sc = torch.zeros_like(trans_sc)
        torsions_sc = torch.zeros(
            num_batch, num_res, 4, device=self._device
        )
        batch = {
            'res_mask': res_mask,
            'diffuse_mask': diffuse_mask,
            'chain_idx': chain_idx,
            'res_idx': res_idx,
            'torsions_sc': torsions_sc,
            'PLM_emb_weight': PLM_embedder._plm_emb_weight,
        }

        if trans_1 is None:
            assert task != 'inverse_folding'
            assert task != 'packing'
            trans_1 = torch.zeros(num_batch, num_res, 3, device=self._device)
        if rotmats_1 is None:
            assert task != 'inverse_folding'
            assert task != 'packing'
            rotmats_1 = torch.eye(3, device=self._device)[None, None].repeat(num_batch, num_res, 1, 1)
        if aatypes_1 is None:
            assert task != 'forward_folding'
            assert task != 'packing'
            aatypes_1 = torch.zeros((num_batch, num_res), device=self._device).long()
        
        logits_1 = torch.nn.functional.one_hot(
            aatypes_1,
            num_classes=self.num_tokens
        ).float()
        if torsions_1 is None:
            torsions_1 = torch.zeros((num_batch, num_res, 4), device=self._device)

        batch['trans_1'] = trans_1
        batch['rotmats_1'] = rotmats_1
        batch['logits_1'] = logits_1
        batch['aatypes_1'] = aatypes_1
        batch['torsions_1'] = torsions_1

        forward_folding = False
        inverse_folding = False
        packing = False
        separate_t = True
        if task.startswith('forward_folding'):
            forward_folding = True
        elif task.startswith('inverse_folding'):
            inverse_folding = True
        elif task.startswith('packing'):
            packing = True

        if forward_folding and separate_t:
            aatypes_0 = aatypes_1
            aatypes_sc = logits_1.clone()

        if inverse_folding and separate_t:
            trans_0 = trans_1
            trans_sc = trans_1
            rotmats_0 = rotmats_1
            rotvecs_sc = so3_utils.rotmat_to_rotvec(rotmats_1)
        
        batch['aatypes_sc'] = aatypes_sc
        batch['trans_sc'] = trans_sc
        batch['rotvecs_sc'] = rotvecs_sc

        # Set-up time
        if num_timesteps is None:
            num_timesteps = self._sample_cfg.num_timesteps

        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps) # from min_t to 1
        t_1 = ts[0]

        # Set-up the initial input
        trans_t_1, rotmats_t_1, aatypes_t_1, torsions_t_1 = trans_0, rotmats_0, aatypes_0, torsions_0
        if forward_folding:
            aatypes_t_1 = aatypes_1
        if inverse_folding:
            trans_t_1 = trans_1
            rotmats_t_1 = rotmats_1
        
        # Set-up the trajectory
        prot_traj = [(
            all_atom.atom37_from_trans_rot(trans_t_1, rotmats_t_1, res_mask).detach().cpu(), 
            aatypes_0.detach().cpu(),
            all_atom.atom37_from_trans_rot_torsion(trans_t_1, rotmats_t_1, torsions_t_1, aatypes_t_1, res_mask).detach().cpu(),   
        )] # *_traj save the data_t in each time step t
        torsions_traj = [torsions_t_1.detach().cpu()]
        clean_traj = [] # *clean_traj save the data_1 in each time step t
        clean_torsions_traj = []

        last_scores = None
        if PLM_encoding is None: # build the template
            with torch.no_grad():
                if PLM_type == 'faESM2_650M' or PLM_type == 'faESMC_600M':
                    # faESM2/faESMC style
                    if PLM_templates is None:
                        max_seqlen = num_res+2
                        batch_chain_lengthes = torch.Tensor([num_res+2,]*num_batch).to(aatypes_t_1.device)
                        cu_seqlens = torch.cumsum(batch_chain_lengthes, dim=0)
                        cu_seqlens = torch.cat([torch.tensor([0]).to(cu_seqlens.device), cu_seqlens], dim=0)
                        ESM_templates = torch.zeros(num_batch, num_res+2).int().to(aatypes_t_1.device)
                        ESM_templates_mask = torch.zeros(num_batch, num_res+2).long().bool().to(aatypes_t_1.device)
                        ESM_templates[:,0] = 21
                        ESM_templates_mask[:, 0] = True
                        ESM_templates[:,-1] = 22
                        ESM_templates_mask[:,-1] = True
                        aatypes_in_ESM_templates = ESM_templates.view(-1).to(aatypes_t_1.dtype)
                    else:
                        max_seqlen = PLM_templates['N_max_res']
                        cu_seqlens = PLM_templates['cu_seqlens']
                        aatypes_in_ESM_templates = PLM_templates['template'].to(aatypes_t_1.dtype)
                        ESM_templates_mask = PLM_templates['template_mask']

                elif PLM_type == 'gLM2_650M':
                    # gLM2 style
                    if PLM_templates is None:
                        gLM_templates = torch.zeros(num_batch, num_res+1).int().to(aatypes_t_1.device)
                        gLM_templates_mask = torch.zeros(num_batch, num_res+1).bool().to(aatypes_t_1.device)
                        gLM_templates[:, 0] = 21
                        gLM_templates_mask[:, 0] = True # for monomer, the first token of each sample is <+>
                        aatypes_in_gLM_templates = gLM_templates.view(-1).to(aatypes_t_1.dtype)
                    else:
                        aatypes_in_gLM_templates = PLM_templates['template'].to(aatypes_t_1.dtype)
                        gLM_templates_mask = PLM_templates['template_mask']
                else:
                    raise ValueError(f'Unsupported PLM type {PLM_type}.')
        # Start sampling
        for t_2 in ts[1:]:

            # Run model.
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            batch['aatypes_t'] = aatypes_t_1
            if forward_folding:
                batch['aatypes_t'] = aatypes_1
            if inverse_folding:
                batch['trans_t'] = trans_1
                batch['rotmats_t'] = rotmats_1
            
            t = torch.ones((num_batch, 1), device=self._device) * t_1
            
            if t_nn is not None:
                batch['r3_t'], batch['so3_t'], batch['cat_t'] = torch.split(t_nn(t), -1)
            else:
                if self._cfg.provide_kappa:
                    batch['so3_t'] = self.rot_sample_kappa(t)
                else:
                    batch['so3_t'] = t
                batch['r3_t'] = t
                batch['cat_t'] = t
                batch['tor_t'] = t 
            
            # TODO: handle torsion in folding or inverse folding
            if forward_folding and separate_t:
                batch['cat_t'] = (1 - self._cfg.min_t) * torch.ones_like(batch['cat_t'])
            if inverse_folding and separate_t:
                batch['r3_t'] = (1 - self._cfg.min_t) * torch.ones_like(batch['r3_t'])
                batch['so3_t'] = (1 - self._cfg.min_t) * torch.ones_like(batch['so3_t'])

            d_t = t_2 - t_1

            if PLM_encoding is None: # encode sequence in real-time
                with torch.no_grad():
                    if PLM_type == 'faESM2_650M':
                        # faESM2 style
                        aatypes_in_faESM = aatypes_in_ESM_templates.clone()
                        aatypes_in_faESM[~ESM_templates_mask.view(-1)] = batch['aatypes_t'].view(-1) # False in template_mask means the non-special tokens
                        plm_s_with_ST = PLM_embedder.faESM2_encoding(aatypes_in_faESM.unsqueeze(0).long(), cu_seqlens.int(), max_seqlen)
                        plm_s = plm_s_with_ST[~ESM_templates_mask.view(-1)].view(*batch['aatypes_t'].shape, PLM_embedder.plm_representations_layer+1, PLM_embedder.plm_representations_dim)
                    elif PLM_type == 'gLM2_650M':
                        # gLM2 style
                        aatypes_in_gLM = aatypes_in_gLM_templates.clone()
                        aatypes_in_gLM[~gLM_templates_mask.view(-1)] = batch['aatypes_t'].view(-1)
                        aatypes_in_gLM = aatypes_in_gLM.view(num_batch, -1)
                        plm_s_with_ST = PLM_embedder.gLM_encoding(aatypes_in_gLM.long())
                        plm_s = plm_s_with_ST[~gLM_templates_mask].view(*batch['aatypes_t'].shape, PLM_embedder.plm_representations_layer+1, PLM_embedder.plm_representations_dim)
                    else:
                        raise ValueError(f'Unsupported PLM type {PLM_type}.')
            else:
                plm_s = PLM_encoding
            plm_s = plm_s.to(batch['trans_t'].dtype)
            batch['PLM_embedding_aatypes_t'] = plm_s

            # run the model
            with torch.no_grad():
                backbone_src = 'backbone'
                if model_order == ['backbone', ]:
                    model_out = model['backbone'](batch)
                else:
                    for model_type in model_order:
                        model_out = None
                        if model_type == 'backbone':
                            model_out = model['backbone'](batch)
                        elif model_type == 'sidechain' and t_1 >= self._cfg.sidechain_start_t:
                            model_out = model['sidechain'](batch)
                            tor_pred=True
                        elif model_type == 'refine' and t_1 >= self._cfg.refine_start_t:
                            # run_refine_score = random.random()
                            # run_refine_score = -0.1
                            # run_threshold = 0.1 + 0.9 * (t_1 - self._cfg.refine_start_t / 1 - self._cfg.refine_start_t)
                            # if run_refine_score < run_threshold:
                            model_out = model['refine'](batch)
                            backbone_src = 'refine'
                        
                        if not model_out is None:
                            batch = batch | model_out
                    
                    model_out = {k: v for k, v in batch.items() if 'pred' in k}

            # Process model output
            pred_trans_1 = model_out[f'backbone_pred_trans']
            pred_rotmats_1 = model_out[f'backbone_pred_rotmats']
            pred_aatypes_1 = model_out[f'backbone_pred_aatypes']
            pred_logits_1 = model_out[f'backbone_pred_logits']

            sc_trans = pred_trans_1
            sc_rotmats = pred_rotmats_1
            sc_logits = pred_logits_1

            if backbone_src == 'refine':

                pred_trans_1 = model_out[f'refine_pred_trans']
                pred_rotmats_1 = model_out[f'refine_pred_rotmats']
                refined_logits_1 = model_out['refine_pred_logits']
                refined_aatypes_1 = model_out[f'refine_pred_aatypes']

                sc_trans = pred_trans_1
                sc_rotmats = pred_rotmats_1
                # sc_logits = refined_logits_1
            
            if tor_pred:
                pred_torsions_1 = model_out['sidechain_pred_torsions']
            else:
                pred_torsions_1 = None
            
            if forward_folding:
                pred_logits_1 = 100.0 * logits_1
            if inverse_folding:
                pred_trans_1 = trans_1
                pred_rotmats_1 = rotmats_1
            if packing:
                pred_logits_1 = 100.0 * logits_1
                pred_trans_1 = trans_1
                pred_rotmats_1 = rotmats_1
                pred_aatypes_1 = aatypes_1

            if self._cfg.self_condition > 0:

                if forward_folding:
                    batch['aatypes_sc'] = logits_1
                    batch['trans_sc'] = _trans_diffuse_mask(
                        sc_trans, trans_1, diffuse_mask)
                    batch['rotvecs_sc'] = _trans_diffuse_mask(
                        so3_utils.rotmat_to_rotvec(sc_rotmats), 
                        so3_utils.rotmat_to_rotvec(rotmats_1),
                        diffuse_mask)
                
                elif inverse_folding:
                    batch['trans_sc'] = trans_1
                    batch['rotvecs_sc'] = so3_utils.rotmat_to_rotvec(rotmats_1)
                    batch['aatypes_sc'] = _trans_diffuse_mask(
                        sc_logits, logits_1, diffuse_mask)
                
                elif packing:
                    batch['aatypes_sc'] = logits_1
                    batch['trans_sc'] = trans_1
                    batch['rotvecs_sc'] = so3_utils.rotmat_to_rotvec(rotmats_1)
                
                else:
                    batch['aatypes_sc'] = _trans_diffuse_mask(
                        sc_logits, logits_1, diffuse_mask)
                    batch['trans_sc'] = _trans_diffuse_mask(
                        sc_trans, trans_1, diffuse_mask)
                    batch['rotvecs_sc'] = _trans_diffuse_mask(
                        so3_utils.rotmat_to_rotvec(sc_rotmats),
                        so3_utils.rotmat_to_rotvec(rotmats_1),
                        diffuse_mask)
                
                # in any situation, groun truth torsion will not be provided
                if tor_pred:
                    # when sidechain is not generated, keep the torsions and torsions_sc as the initial state
                    batch['torsions_sc'] = _torsions_diffuse_mask(pred_torsions_1, torsions_1, diffuse_mask)

            if tor_pred:
                clean_traj.append((
                    all_atom.atom37_from_trans_rot(pred_trans_1, pred_rotmats_1, res_mask).detach().cpu(), 
                    pred_aatypes_1.detach().cpu(),
                    all_atom.atom37_from_trans_rot_torsion(pred_trans_1, pred_rotmats_1, pred_torsions_1, pred_aatypes_1, res_mask).detach().cpu(), 
                ))
                clean_torsions_traj.append(pred_torsions_1.detach().cpu())
            else:
                clean_traj.append((
                    all_atom.atom37_from_trans_rot(pred_trans_1, pred_rotmats_1, res_mask).detach().cpu(),
                    pred_aatypes_1.detach().cpu(),))
            
            # Take reverse step
            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)
            if rot_style == 'multiflow':            
                rotmats_t_2 = self._rots_euler_step(
                    d_t, t_1, pred_rotmats_1, rotmats_t_1)
            elif rot_style == 'foldflow':
                rot_vectorfield = model_out['rot_vectorfield']
                rotmats_t_2 = foldflow_reverse(rot_t=rotmats_t_1,
                                               v_t=rot_vectorfield,
                                               t=t_1,
                                               dt=d_t,
                                               noise_scale=1.0)
            else:
                raise ValueError(f'{rot_style} has not been implemented')

            if not pred_torsions_1 is None:
                torsions_t_2 = self._torsions_euler_step(d_t, t_1, pred_torsions_1, torsions_t_1)
            else:
                torsions_t_2 = torsions_0

            curr_t = max(1e-4, min(t_2.item(), 1-1e-4))

            if curr_t >= 0.85:
                argmax=True
            else:
                argmax=False

            curr_temp = exp_schedule(curr_t, self._aatypes_cfg.max_temp, self._aatypes_cfg.decay_rate)

            if backbone_src == 'refine':
                aatypes_t_2, aatypes_t_2_scores = self._aatypes_euler_step_APM_Refine(d_t, t_1, pred_logits_1, refined_logits_1, noise_scale=0, temperature=curr_temp, argmax=argmax)
            else:
                aatypes_t_2, aatypes_t_2_scores = self._aatypes_euler_step_APM(d_t, t_1, pred_logits_1, noise_scale=1.0-curr_t, temperature=curr_temp, argmax=argmax, diffuse_mask=diffuse_mask, apply_penalty=self._aatypes_cfg.apply_penalty)

            trans_t_2 = _trans_diffuse_mask(trans_t_2, trans_1, diffuse_mask)
            rotmats_t_2 = _rots_diffuse_mask(rotmats_t_2, rotmats_1, diffuse_mask)
            aatypes_t_2 = _aatypes_diffuse_mask(aatypes_t_2, aatypes_1, diffuse_mask)
            torsions_t_2 = _torsions_diffuse_mask(torsions_t_2, torsions_1, diffuse_mask)
            trans_t_1, rotmats_t_1, aatypes_t_1, torsions_t_1 = trans_t_2, rotmats_t_2, aatypes_t_2, torsions_t_2
            
            if tor_pred:
                prot_traj.append((
                    all_atom.atom37_from_trans_rot(trans_t_1, rotmats_t_1, res_mask).detach().cpu(), 
                    aatypes_0.detach().cpu(),
                    all_atom.atom37_from_trans_rot_torsion(trans_t_1, rotmats_t_1, torsions_t_1, aatypes_t_1, res_mask).detach().cpu(),   
                ))
                torsions_traj.append(torsions_t_2.cpu().detach())
            else:
                prot_traj.append((
                    all_atom.atom37_from_trans_rot(trans_t_1, rotmats_t_1, res_mask).detach().cpu(),
                    aatypes_0.detach().cpu(),))

            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['aatypes_t'] = aatypes_t_1
        if forward_folding:
            batch['aatypes_t'] = aatypes_1
        if inverse_folding:
            batch['trans_t'] = trans_1
            batch['rotmats_t'] = rotmats_1
        
        if PLM_encoding is None: # encode sequence in real-time
            with torch.no_grad():
                if PLM_type == 'faESM2_650M':
                    # faESM2 style
                    aatypes_in_faESM = aatypes_in_ESM_templates.clone()
                    aatypes_in_faESM[~ESM_templates_mask.view(-1)] = batch['aatypes_t'].view(-1) # False in template_mask means the non-special tokens
                    plm_s_with_ST = PLM_embedder.faESM2_encoding(aatypes_in_faESM.unsqueeze(0).long(), cu_seqlens.int(), max_seqlen)
                    plm_s = plm_s_with_ST[~ESM_templates_mask.view(-1)].view(*batch['aatypes_t'].shape, PLM_embedder.plm_representations_layer+1, PLM_embedder.plm_representations_dim)
                elif PLM_type == 'gLM2_650M':
                    # gLM2 style
                    aatypes_in_gLM = aatypes_in_gLM_templates.clone()
                    aatypes_in_gLM[~gLM_templates_mask.view(-1)] = batch['aatypes_t'].view(-1)
                    aatypes_in_gLM = aatypes_in_gLM.view(num_batch, -1)
                    plm_s_with_ST = PLM_embedder.gLM_encoding(aatypes_in_gLM.long())
                    plm_s = plm_s_with_ST[~gLM_templates_mask].view(*batch['aatypes_t'].shape, PLM_embedder.plm_representations_layer+1, PLM_embedder.plm_representations_dim)
                else:
                    raise ValueError(f'Unsupported PLM type {PLM_type}.')
        else:
            plm_s = PLM_encoding
        plm_s = plm_s.to(batch['trans_t'].dtype)
        batch['PLM_embedding_aatypes_t'] = plm_s

        with torch.no_grad():
            backbone_src = 'backbone'
            tor_pred = False
            if model_order == ['backbone', ]:
                model_out = model['backbone'](batch)
            else:
                for model_type in model_order:
                    model_out = None
                    if model_type == 'backbone':
                        model_out = model['backbone'](batch)
                    elif model_type == 'sidechain':
                        model_out = model['sidechain'](batch)
                        tor_pred=True
                    elif model_type == 'refine':
                        model_out = model['refine'](batch)
                        backbone_src = 'refine'
                    
                    if not model_out is None:
                        batch = batch | model_out
                
                model_out = {k: v for k, v in batch.items() if 'pred' in k}

        pred_trans_1 = model_out[f'backbone_pred_trans']
        pred_rotmats_1 = model_out[f'backbone_pred_rotmats']
        pred_aatypes_1 = model_out[f'backbone_pred_aatypes']
        pred_logits_1 = model_out[f'backbone_pred_logits']
        if backbone_src == 'refine':
            pred_trans_1 = model_out[f'refine_pred_trans']
            pred_rotmats_1 = model_out[f'refine_pred_rotmats']
            # pred_aatypes_1 = model_out[f'refine_pred_aatypes']
            refined_logits_1 = model_out['refine_pred_logits']
            pred_aatypes_1, aatypes_t_2_scores = self._aatypes_euler_step_APM_Refine(d_t, t_1, pred_logits_1, refined_logits_1, noise_scale=0, temperature=curr_temp, argmax=True)

        if tor_pred:
            pred_torsions_1 = model_out['sidechain_pred_torsions']
        else:
            pred_torsions_1 = None

        if forward_folding:
            pred_aatypes_1 = aatypes_1
        if inverse_folding:
            pred_trans_1 = trans_1
            pred_rotmats_1 = rotmats_1

        pred_atom37 = all_atom.atom37_from_trans_rot(pred_trans_1, pred_rotmats_1, res_mask).detach().cpu()
        last_step = [pred_atom37, pred_aatypes_1.detach().cpu(),]
        if tor_pred:
            pred_atom37_Full = all_atom.atom37_from_trans_rot_torsion(pred_trans_1, pred_rotmats_1, pred_torsions_1, pred_aatypes_1, res_mask).detach().cpu()
            last_step.append(pred_atom37_Full)
            clean_torsions_traj.append(pred_torsions_1.detach().cpu())
            torsions_traj.append(pred_torsions_1.detach().cpu())
        
        clean_traj.append(last_step)
        prot_traj.append(last_step)
        
        gt_result = {
            'gt_atom37': all_atom.atom37_from_trans_rot(trans_1, rotmats_1, res_mask).detach().cpu(),
            'gt_aatypes': aatypes_1,
            'gt_atom37_full': all_atom.atom37_from_trans_rot_torsion(trans_1, rotmats_1, torsions_1, aatypes_1, res_mask).detach().cpu(),
        }

        return prot_traj, clean_traj, torsions_traj, clean_torsions_traj, gt_result

    def sample_chain_by_chain(
            self,
            num_batch,
            num_res,
            model,
            model_order, 
            num_timesteps=None,
            trans_0=None,
            rotmats_0=None,
            aatypes_0=None,
            torsions_0=None,
            trans_1=None,
            rotmats_1=None,
            aatypes_1=None,
            torsions_1=None,
            diffuse_mask_gen=None,
            chain_idx=None,
            res_idx=None,
            rot_style='multiflow',
            PLM_embedder=None,
            PLM_type=None, 
            PLM_encoding=None, 
            PLM_templates=None
        ):

        tor_pred = False
        reset_center = False
        gen_num_res = int(diffuse_mask_gen[0].sum().cpu().item())
        fix_num_res = diffuse_mask_gen.shape[1] - gen_num_res
        do_reset = fix_num_res > 0
        
        res_mask = torch.ones(num_batch, num_res, device=self._device)
        diffuse_mask = torch.ones(num_batch, num_res, device=self._device)

        trans_sc_init = torch.zeros(num_batch, gen_num_res, 3, device=self._device)
        aatypes_sc_init = torch.zeros(
            num_batch, gen_num_res, self.num_tokens, device=self._device)
        rotvecs_sc_init = torch.zeros_like(trans_sc_init)
        torsions_sc_init = torch.zeros(
            num_batch, num_res, 4, device=self._device
        )
        batch = {
            'res_mask': res_mask,
            'diffuse_mask': diffuse_mask,
            'chain_idx': chain_idx,
            'res_idx': res_idx,
            'PLM_emb_weight': PLM_embedder._plm_emb_weight,
        }

        if trans_1 is None:
            trans_1 = torch.zeros(num_batch, num_res, 3, device=self._device)
            batch['trans_1'] = trans_1
        else:
            trans_1_comb = torch.cat([trans_1, torch.zeros(num_batch, gen_num_res, 3, device=self._device)], dim=1)
            batch['trans_1'] = trans_1_comb
        if rotmats_1 is None:
            rotmats_1 = torch.eye(3, device=self._device)[None, None].repeat(num_batch, num_res, 1, 1)
            batch['rotmats_1'] = rotmats_1
        else:
            rotmats_1_comb = torch.cat([rotmats_1, torch.eye(3, device=self._device)[None, None].repeat(num_batch, gen_num_res, 1, 1)], dim=1)
            batch['rotmats_1'] = rotmats_1_comb
        if aatypes_1 is None:
            aatypes_1 = torch.zeros((num_batch, num_res), device=self._device).long()
            batch['aatypes_1'] = aatypes_1
            logits_1 = torch.nn.functional.one_hot(
                aatypes_1.long(),
                num_classes=self.num_tokens
            ).float()
            logits_1 = logits_1 * 100
            batch['logits_1'] = logits_1
        else:
            init_aatypes_1 = torch.zeros(num_batch, gen_num_res, device=self._device).long()
            aatypes_1_comb = torch.cat([aatypes_1, init_aatypes_1], dim=1)
            batch['aatypes_1'] = aatypes_1_comb.long()
            logits_1 = torch.nn.functional.one_hot(
                aatypes_1.long(),
                num_classes=self.num_tokens
            ).float()
            logits_1 = logits_1 * 100
            init_logits_1 = torch.nn.functional.one_hot(
                init_aatypes_1.long(),
                num_classes=self.num_tokens
            ).float()
            init_logits_1 = init_logits_1 * 100
            logits_1_comb = torch.cat([logits_1, init_logits_1], dim=1)

            batch['logits_1'] = logits_1_comb


        
        # logits_1 = torch.nn.functional.one_hot(
        #     aatypes_1.long(),
        #     num_classes=self.num_tokens
        # ).float()
        # logits_1 = logits_1 * 100

        # batch['trans_1'] = trans_1
        # batch['rotmats_1'] = rotmats_1
        # batch['logits_1'] = logits_1
        # batch['aatypes_1'] = aatypes_1
        torsions_1 = torch.zeros((num_batch, num_res, 4), device=self._device)
        batch['torsions_1'] = torsions_1

        forward_folding = False
        inverse_folding = False
        
        if do_reset:
            trans_sc = torch.cat([trans_1, trans_sc_init], dim=1)
            rotvecs_sc = torch.cat([so3_utils.rotmat_to_rotvec(rotmats_1), rotvecs_sc_init], dim=1)
            aatypes_sc = torch.cat([logits_1, aatypes_sc_init], dim=1)
        else:
            trans_sc = trans_sc_init
            rotvecs_sc = rotvecs_sc_init
            aatypes_sc = aatypes_sc_init

        ### if fixed sidechains, open this
        # torsions_sc = _trans_diffuse_mask(torsions_sc_init, torsions_1, diffuse_mask_gen)
        torsions_sc = torsions_sc_init

        batch['aatypes_sc'] = aatypes_sc
        batch['trans_sc'] = trans_sc
        batch['rotvecs_sc'] = rotvecs_sc
        batch['torsions_sc'] = torsions_sc

        # Set-up time
        if num_timesteps is None:
            num_timesteps = self._sample_cfg.num_timesteps

        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps) # from min_t to 1
        t_1 = ts[0]

        # Set-up the initial input
        trans_t_1, rotmats_t_1, aatypes_t_1, torsions_t_1 = trans_0, rotmats_0, aatypes_0, torsions_0
        
        # Set-up the trajectory
        prot_traj = [(
            all_atom.atom37_from_trans_rot(trans_t_1, rotmats_t_1, res_mask).detach().cpu(), 
            aatypes_0.detach().cpu(),
            all_atom.atom37_from_trans_rot_torsion(trans_t_1, rotmats_t_1, torsions_t_1, aatypes_t_1, res_mask).detach().cpu(),   
        )] # *_traj save the data_t in each time step t
        torsions_traj = [torsions_t_1.detach().cpu()]
        clean_traj = [] # *clean_traj save the data_1 in each time step t
        clean_torsions_traj = []

        if PLM_encoding is None: # build the template
            with torch.no_grad():
                if PLM_type == 'faESM2_650M' or PLM_type == 'faESMC_600M':
                    # faESM2/faESMC style
                    if PLM_templates is None:
                        max_seqlen = num_res+2
                        batch_chain_lengthes = torch.Tensor([num_res+2,]*num_batch).to(aatypes_t_1.device)
                        cu_seqlens = torch.cumsum(batch_chain_lengthes, dim=0)
                        cu_seqlens = torch.cat([torch.tensor([0]).to(cu_seqlens.device), cu_seqlens], dim=0)
                        ESM_templates = torch.zeros(num_batch, num_res+2).int().to(aatypes_t_1.device)
                        ESM_templates_mask = torch.zeros(num_batch, num_res+2).long().bool().to(aatypes_t_1.device)
                        ESM_templates[:,0] = 21
                        ESM_templates_mask[:, 0] = True
                        ESM_templates[:,-1] = 22
                        ESM_templates_mask[:,-1] = True
                        aatypes_in_ESM_templates = ESM_templates.view(-1).to(aatypes_t_1.dtype)
                    else:
                        max_seqlen = PLM_templates['N_max_res']
                        cu_seqlens = PLM_templates['cu_seqlens']
                        aatypes_in_ESM_templates = PLM_templates['template'].to(aatypes_t_1.dtype)
                        ESM_templates_mask = PLM_templates['template_mask']

                elif PLM_type == 'gLM2_650M':
                    # gLM2 style
                    if PLM_templates is None:
                        gLM_templates = torch.zeros(num_batch, num_res+1).int().to(aatypes_t_1.device)
                        gLM_templates_mask = torch.zeros(num_batch, num_res+1).bool().to(aatypes_t_1.device)
                        gLM_templates[:, 0] = 21
                        gLM_templates_mask[:, 0] = True # for monomer, the first token of each sample is <+>
                        aatypes_in_gLM_templates = gLM_templates.view(-1).to(aatypes_t_1.dtype)
                    else:
                        aatypes_in_gLM_templates = PLM_templates['template'].to(aatypes_t_1.dtype)
                        gLM_templates_mask = PLM_templates['template_mask']
                else:
                    raise ValueError(f'Unsupported PLM type {PLM_type}.')
        # Start sampling
        for t_2 in ts[1:]:
            
            t = torch.ones((num_batch, 1), device=self._device) * t_1

            if do_reset and t_1.item() >= 1.1 and not reset_center:
                reset_center = True
                # trans_t_1 = trans_t_1 - trans_t_1.mean(dim=1, keepdim=True)
                diffuse_mask = torch.ones_like(diffuse_mask)

                fix_trans = trans_t_1[:, :fix_num_res]
                # print('#'*10, fix_trans.shape)
                fix_rots = rotmats_t_1[:, :fix_num_res]
                # print('#'*10, fix_rots.shape)
                fix_aatypes = aatypes_t_1[:, :fix_num_res]
                fix_logits = torch.nn.functional.one_hot(
                    fix_aatypes.long(),
                    num_classes=self.num_tokens
                ).float() * 100.0
                fix_mask = torch.ones_like(fix_aatypes).float()

                noised_trans, noised_rots = self._corrupt_se3(fix_trans, t, fix_rots, t, res_mask=fix_mask, diffuse_mask=fix_mask, rot_style=rot_style)
                noised_aatypes = self._corrupt_aatypes(fix_aatypes, t, res_mask=fix_mask, diffuse_mask=fix_mask)

                gen_trans = trans_t_1[:, fix_num_res:]
                gen_rots = rotmats_t_1[:, fix_num_res:]
                gen_aatypes = aatypes_t_1[:, fix_num_res:]

                trans_t_1 = torch.cat([noised_trans, gen_trans], dim=1)
                rotmats_t_1 = torch.cat([noised_rots, gen_rots], dim=1)
                aatypes_t_1 = torch.cat([noised_aatypes, gen_aatypes], dim=1)

                batch['trans_sc'] = torch.cat([fix_trans, batch['trans_sc'][:, fix_num_res:, :]], dim=1)
                batch['rotvecs_sc'] = torch.cat([so3_utils.rotmat_to_rotvec(fix_rots), batch['rotvecs_sc'][:, fix_num_res:, :]], dim=1)
                batch['aatypes_sc'] = torch.cat([fix_logits, batch['aatypes_sc'][:, fix_num_res:, :]], dim=1)

            # Run model.
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            batch['aatypes_t'] = aatypes_t_1.int()

            # print('#'*10, batch['aatypes_t'].dtype)
            
            # t = torch.ones((num_batch, 1), device=self._device) * t_1

            # t = (1 - self._cfg.min_t) * torch.ones_like(diffuse_mask) * (1- diffuse_mask) + t * diffuse_mask
            # t = torch.ones_like(diffuse_mask) * (1- diffuse_mask) + t * diffuse_mask

            if self._cfg.provide_kappa:
                batch['so3_t'] = self.rot_sample_kappa(t)
            else:
                batch['so3_t'] = t
            batch['r3_t'] = t
            batch['cat_t'] = t
            batch['tor_t'] = t 
            
            d_t = t_2 - t_1

            if PLM_encoding is None: # encode sequence in real-time
                with torch.no_grad():
                    if PLM_type == 'faESM2_650M':
                        # faESM2 style
                        aatypes_in_faESM = aatypes_in_ESM_templates.clone()
                        aatypes_in_faESM[~ESM_templates_mask.view(-1)] = batch['aatypes_t'].view(-1) # False in template_mask means the non-special tokens
                        plm_s_with_ST = PLM_embedder.faESM2_encoding(aatypes_in_faESM.unsqueeze(0).long(), cu_seqlens.int(), max_seqlen)
                        plm_s = plm_s_with_ST[~ESM_templates_mask.view(-1)].view(*batch['aatypes_t'].shape, PLM_embedder.plm_representations_layer+1, PLM_embedder.plm_representations_dim)
                    elif PLM_type == 'gLM2_650M':
                        # gLM2 style
                        aatypes_in_gLM = aatypes_in_gLM_templates.clone()
                        aatypes_in_gLM[~gLM_templates_mask.view(-1)] = batch['aatypes_t'].view(-1)
                        aatypes_in_gLM = aatypes_in_gLM.view(num_batch, -1)
                        plm_s_with_ST = PLM_embedder.gLM_encoding(aatypes_in_gLM.long())
                        plm_s = plm_s_with_ST[~gLM_templates_mask].view(*batch['aatypes_t'].shape, PLM_embedder.plm_representations_layer+1, PLM_embedder.plm_representations_dim)
                    else:
                        raise ValueError(f'Unsupported PLM type {PLM_type}.')
            else:
                plm_s = PLM_encoding
            plm_s = plm_s.to(batch['trans_t'].dtype)
            batch['PLM_embedding_aatypes_t'] = plm_s

            # run the model
            with torch.no_grad():
                backbone_src = 'backbone'
                if model_order == ['backbone', ]:
                    model_out = model['backbone'](batch)
                else:
                    for model_type in model_order:
                        model_out = None
                        if model_type == 'backbone':
                            model_out = model['backbone'](batch)
                            # if do_reset:
                            #     model_out['backbone_pred_trans'] = _trans_diffuse_mask(model_out['backbone_pred_trans'], trans_1, diffuse_mask)
                            #     model_out['backbone_pred_rotmats'] = _rots_diffuse_mask(model_out['backbone_pred_rotmats'], rotmats_1, diffuse_mask)
                            #     model_out['backbone_pred_aatypes'] = _aatypes_diffuse_mask(model_out[f'backbone_pred_aatypes'], aatypes_1, diffuse_mask)
                        elif model_type == 'sidechain' and t_1 >= self._cfg.sidechain_start_t:
                            model_out = model['sidechain'](batch)
                            tor_pred=True
                        elif model_type == 'refine' and t_1 >= self._cfg.refine_start_t:
                            model_out = model['refine'](batch)
                            backbone_src = 'refine'
                        
                        if not model_out is None:
                            batch = batch | model_out
                    
                    model_out = {k: v for k, v in batch.items() if 'pred' in k}

            # Process model output
            pred_trans_1 = model_out[f'backbone_pred_trans']
            pred_rotmats_1 = model_out[f'backbone_pred_rotmats']
            pred_aatypes_1 = model_out[f'backbone_pred_aatypes']
            pred_logits_1 = model_out[f'backbone_pred_logits']

            sc_trans = pred_trans_1
            sc_rotmats = pred_rotmats_1
            sc_logits = pred_logits_1

            if backbone_src == 'refine':

                pred_trans_1 = model_out[f'refine_pred_trans']
                pred_rotmats_1 = model_out[f'refine_pred_rotmats']
                refined_logits_1 = model_out['refine_pred_logits']

                sc_trans = pred_trans_1
                sc_rotmats = pred_rotmats_1
                sc_logits = pred_logits_1 * 0.8 + refined_logits_1 * 0.2
            
            if tor_pred:
                pred_torsions_1 = model_out['sidechain_pred_torsions']
            else:
                pred_torsions_1 = None
            
            # if forward_folding:
            #     pred_logits_1 = 100.0 * logits_1
            # if inverse_folding:
            #     pred_trans_1 = trans_1
            #     pred_rotmats_1 = rotmats_1
            # if packing:
            #     pred_logits_1 = 100.0 * logits_1
            #     pred_trans_1 = trans_1
            #     pred_rotmats_1 = rotmats_1
            #     pred_aatypes_1 = aatypes_1

            if self._cfg.self_condition > 0:
                if do_reset:
                    curr_logits_1 = logits_1 * (1-t) + sc_logits[:, :fix_num_res] * t
                    batch['aatypes_sc'] = torch.cat([curr_logits_1, sc_logits[:, fix_num_res:]], dim=1)
                    curr_trans_1 = trans_1 * (1-t) + sc_trans[:, :fix_num_res] * t
                    batch['trans_sc'] = torch.cat([curr_trans_1, sc_trans[:, fix_num_res:]], dim=1)
                    rot_vecs_1 = so3_utils.rotmat_to_rotvec(rotmats_1)
                    rot_vecs_pred = so3_utils.rotmat_to_rotvec(sc_rotmats)
                    curr_rotvecs_1 = rot_vecs_1 * (1-t) + rot_vecs_pred[:, :fix_num_res] * t
                    batch['rotvecs_sc'] = torch.cat([curr_rotvecs_1, rot_vecs_pred[:, fix_num_res:]], dim=1)
                else:
                    batch['aatypes_sc'] = sc_logits
                    batch['trans_sc'] = sc_trans
                    batch['rotvecs_sc'] = so3_utils.rotmat_to_rotvec(sc_rotmats)
                    # batch['aatypes_sc'] = _trans_diffuse_mask(
                    #     sc_logits, logits_1, diffuse_mask)
                    # batch['trans_sc'] = _trans_diffuse_mask(
                    #     sc_trans, trans_1, diffuse_mask)
                    # batch['rotvecs_sc'] = _trans_diffuse_mask(
                    #     so3_utils.rotmat_to_rotvec(sc_rotmats),
                    #     so3_utils.rotmat_to_rotvec(rotmats_1),
                    #     diffuse_mask)
                
                # in any situation, groun truth torsion will not be provided
                if tor_pred:
                    # when sidechain is not generated, keep the torsions and torsions_sc as the initial state
                    ### if fixed sidechains, open this
                    # batch['torsions_sc'] = _torsions_diffuse_mask(pred_torsions_1, torsions_1, diffuse_mask)
                    batch['torsions_sc'] = pred_torsions_1

            if tor_pred:
                clean_traj.append((
                    all_atom.atom37_from_trans_rot(pred_trans_1, pred_rotmats_1, res_mask).detach().cpu(), 
                    pred_aatypes_1.detach().cpu(),
                    all_atom.atom37_from_trans_rot_torsion(pred_trans_1, pred_rotmats_1, pred_torsions_1, pred_aatypes_1, res_mask).detach().cpu(), 
                ))
                clean_torsions_traj.append(pred_torsions_1.detach().cpu())
            else:
                clean_traj.append((
                    all_atom.atom37_from_trans_rot(pred_trans_1, pred_rotmats_1, res_mask).detach().cpu(),
                    pred_aatypes_1.detach().cpu(),))
            
            # Take reverse step
            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)
            if rot_style == 'multiflow':            
                rotmats_t_2 = self._rots_euler_step(
                    d_t, t_1, pred_rotmats_1, rotmats_t_1)
            elif rot_style == 'foldflow':
                rot_vectorfield = model_out['rot_vectorfield']
                rotmats_t_2 = foldflow_reverse(rot_t=rotmats_t_1,
                                               v_t=rot_vectorfield,
                                               t=t_1,
                                               dt=d_t,
                                               noise_scale=1.0)
            else:
                raise ValueError(f'{rot_style} has not been implemented')

            if not pred_torsions_1 is None:
                torsions_t_2 = self._torsions_euler_step(d_t, t_1, pred_torsions_1, torsions_t_1)
            else:
                torsions_t_2 = torsions_0

            curr_t = max(1e-4, min(t_2.item(), 1-1e-4))

            # segment 2
            curr_temp = (-1 * np.log(curr_t)+ 1e-4) ** 3
            if curr_t >= 0.85:
                argmax=True
            else:
                argmax=False

            if backbone_src == 'refine':
                aatypes_t_2, aatypes_t_2_scores = self._aatypes_euler_step_APM_Refine(d_t, t_1, pred_logits_1, refined_logits_1, noise_scale=0, temperature=curr_temp, argmax=argmax, diffuse_mask=diffuse_mask)
                # aatypes_t_2, aatypes_t_2_scores = self._aatypes_euler_step_APM_Refine_guide(d_t, t_1, pred_logits_1, refined_logits_1, noise_scale=0, temperature=curr_temp, argmax=argmax)
            else:
                aatypes_t_2, aatypes_t_2_scores = self._aatypes_euler_step_APM(d_t, t_1, pred_logits_1, noise_scale=1.0-curr_t, temperature=curr_temp, argmax=argmax, diffuse_mask=diffuse_mask)

            # trans_t_2 = _trans_diffuse_mask(trans_t_2, trans_1, diffuse_mask)
            # rotmats_t_2 = _rots_diffuse_mask(rotmats_t_2, rotmats_1, diffuse_mask)
            # aatypes_t_2 = _aatypes_diffuse_mask(aatypes_t_2, aatypes_1, diffuse_mask)
            ### if fixed sidechains, open this
            # torsions_t_2 = _torsions_diffuse_mask(torsions_t_2, torsions_1, diffuse_mask)
            trans_t_1, rotmats_t_1, aatypes_t_1, torsions_t_1 = trans_t_2, rotmats_t_2, aatypes_t_2, torsions_t_2
            
            if tor_pred:
                prot_traj.append((
                    all_atom.atom37_from_trans_rot(trans_t_1, rotmats_t_1, res_mask).detach().cpu(), 
                    aatypes_0.detach().cpu(),
                    all_atom.atom37_from_trans_rot_torsion(trans_t_1, rotmats_t_1, torsions_t_1, aatypes_t_1, res_mask).detach().cpu(),   
                ))
                torsions_traj.append(torsions_t_2.cpu().detach())
            else:
                prot_traj.append((
                    all_atom.atom37_from_trans_rot(trans_t_1, rotmats_t_1, res_mask).detach().cpu(),
                    aatypes_0.detach().cpu(),))

            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['aatypes_t'] = aatypes_t_1.int()
        if forward_folding:
            batch['aatypes_t'] = aatypes_1
        if inverse_folding:
            batch['trans_t'] = trans_1
            batch['rotmats_t'] = rotmats_1
        
        if PLM_encoding is None: # encode sequence in real-time
            with torch.no_grad():
                if PLM_type == 'faESM2_650M':
                    # faESM2 style
                    aatypes_in_faESM = aatypes_in_ESM_templates.clone()
                    aatypes_in_faESM[~ESM_templates_mask.view(-1)] = batch['aatypes_t'].view(-1) # False in template_mask means the non-special tokens
                    plm_s_with_ST = PLM_embedder.faESM2_encoding(aatypes_in_faESM.unsqueeze(0).long(), cu_seqlens.int(), max_seqlen)
                    plm_s = plm_s_with_ST[~ESM_templates_mask.view(-1)].view(*batch['aatypes_t'].shape, PLM_embedder.plm_representations_layer+1, PLM_embedder.plm_representations_dim)
                elif PLM_type == 'gLM2_650M':
                    # gLM2 style
                    aatypes_in_gLM = aatypes_in_gLM_templates.clone()
                    aatypes_in_gLM[~gLM_templates_mask.view(-1)] = batch['aatypes_t'].view(-1)
                    aatypes_in_gLM = aatypes_in_gLM.view(num_batch, -1)
                    plm_s_with_ST = PLM_embedder.gLM_encoding(aatypes_in_gLM.long())
                    plm_s = plm_s_with_ST[~gLM_templates_mask].view(*batch['aatypes_t'].shape, PLM_embedder.plm_representations_layer+1, PLM_embedder.plm_representations_dim)
                else:
                    raise ValueError(f'Unsupported PLM type {PLM_type}.')
        else:
            plm_s = PLM_encoding
        plm_s = plm_s.to(batch['trans_t'].dtype)
        batch['PLM_embedding_aatypes_t'] = plm_s

        with torch.no_grad():
            backbone_src = 'backbone'
            tor_pred = False
            if model_order == ['backbone', ]:
                model_out = model['backbone'](batch)
            else:
                for model_type in model_order:
                    model_out = None
                    if model_type == 'backbone':
                        model_out = model['backbone'](batch)
                        # if do_reset:
                        #     model_out['backbone_pred_trans'] = _trans_diffuse_mask(model_out['backbone_pred_trans'], trans_1, diffuse_mask)
                        #     model_out['backbone_pred_rotmats'] = _rots_diffuse_mask(model_out['backbone_pred_rotmats'], rotmats_1, diffuse_mask)
                        #     model_out['backbone_pred_aatypes'] = _aatypes_diffuse_mask(model_out[f'backbone_pred_aatypes'], aatypes_1, diffuse_mask)
                    elif model_type == 'sidechain':
                        model_out = model['sidechain'](batch)
                        tor_pred=True
                    elif model_type == 'refine':
                        model_out = model['refine'](batch)
                        backbone_src = 'refine'
                    
                    if not model_out is None:
                        batch = batch | model_out
                
                model_out = {k: v for k, v in batch.items() if 'pred' in k}

        pred_trans_1 = model_out[f'backbone_pred_trans']
        pred_rotmats_1 = model_out[f'backbone_pred_rotmats']
        pred_aatypes_1 = model_out[f'backbone_pred_aatypes']
        pred_logits_1 = model_out[f'backbone_pred_logits']
        if backbone_src == 'refine':
            pred_trans_1 = model_out[f'refine_pred_trans']
            pred_rotmats_1 = model_out[f'refine_pred_rotmats']
            refined_logits_1 = model_out['refine_pred_logits']
            pred_aatypes_1, aatypes_t_2_scores = self._aatypes_euler_step_APM_Refine(d_t, t_1, pred_logits_1, refined_logits_1, noise_scale=0, temperature=curr_temp, argmax=True, diffuse_mask=diffuse_mask)

        if tor_pred:
            pred_torsions_1 = model_out['sidechain_pred_torsions']
        else:
            pred_torsions_1 = None


        # pred_trans_1 = _trans_diffuse_mask(pred_trans_1, trans_1, diffuse_mask)
        # pred_rotmats_1 = _rots_diffuse_mask(pred_rotmats_1, rotmats_1, diffuse_mask)
        # pred_aatypes_1 = _aatypes_diffuse_mask(pred_aatypes_1, aatypes_1, diffuse_mask)
        ### if fixed sidechains, open this
        # if tor_pred:
        #     pred_torsions_1 = _torsions_diffuse_mask(pred_torsions_1, torsions_1, diffuse_mask)

        pred_atom37 = all_atom.atom37_from_trans_rot(pred_trans_1, pred_rotmats_1, res_mask).detach().cpu()
        last_step = [pred_atom37, pred_aatypes_1.detach().cpu(),]
        if tor_pred:
            pred_atom37_Full = all_atom.atom37_from_trans_rot_torsion(pred_trans_1, pred_rotmats_1, pred_torsions_1, pred_aatypes_1, res_mask).detach().cpu()
            last_step.append(pred_atom37_Full)
            clean_torsions_traj.append(pred_torsions_1.detach().cpu())
            torsions_traj.append(pred_torsions_1.detach().cpu())
        
        clean_traj.append(last_step)
        prot_traj.append(last_step)
        
        if do_reset:
            gt_result = {
            'gt_atom37': all_atom.atom37_from_trans_rot(trans_1_comb, rotmats_1_comb, res_mask).detach().cpu(),
            'gt_aatypes': aatypes_1_comb,
            'gt_atom37_full': all_atom.atom37_from_trans_rot_torsion(trans_1_comb, rotmats_1_comb, torsions_1, aatypes_1_comb, res_mask).detach().cpu(),
            }
        else:
            gt_result = {
                'gt_atom37': all_atom.atom37_from_trans_rot(trans_1, rotmats_1, res_mask).detach().cpu(),
                'gt_aatypes': aatypes_1,
                'gt_atom37_full': all_atom.atom37_from_trans_rot_torsion(trans_1, rotmats_1, torsions_1, aatypes_1, res_mask).detach().cpu(),
            }

        pred = [pred_trans_1, pred_rotmats_1, pred_aatypes_1]
        if tor_pred:
            pred.append(pred_torsions_1)

        return prot_traj, clean_traj, torsions_traj, clean_torsions_traj, gt_result, pred


    def sample_conditional(
            self,
            num_batch,
            num_res,
            model,
            model_order, 
            num_timesteps=None,
            trans_0=None,
            rotmats_0=None,
            aatypes_0=None,
            torsions_0=None,
            trans_1=None,
            rotmats_1=None,
            aatypes_1=None,
            torsions_1=None,
            diffuse_mask=None,
            chain_idx=None,
            res_idx=None,
            t_nn=None,
            forward_folding=False,
            inverse_folding=False,
            packing=False,
            separate_t=False,
            rot_style='multiflow',
            PLM_embedder=None,
            PLM_type=None, 
            PLM_encoding=None, 
            PLM_templates=None,
            res_mask=None,
        ):

        tor_pred = False
        
        res_mask = torch.ones(num_batch, num_res, device=self._device) if res_mask is None else res_mask

        # Set-up initial prior samples
        if trans_0 is None:
            trans_0 = _centered_gaussian(
                num_batch, num_res, self._device) * du.NM_TO_ANG_SCALE
        if rotmats_0 is None:
            # rotmats_0 = _uniform_so3(num_batch, num_res, self._device)
            rotmats_0 = self.sample_rand_rot(num_batch, num_res, dtype=torch.float)
        if torsions_0 is None:
            torsions_0 = _uniform_torsion(num_batch, num_res, self._device)
        if aatypes_0 is None:
            # according to the previous results, we only use masking noise for aatypes
            aatypes_0 = _masked_categorical(num_batch, num_res, self._device)
        if res_idx is None:
            res_idx = torch.arange(
                num_res,
                device=self._device,
                dtype=torch.float32)[None].repeat(num_batch, 1)
            res_idx += 1 # index in traning processing is 1-based
        
        if chain_idx is None:
            chain_idx = res_mask

        if diffuse_mask is None:
            diffuse_mask = res_mask

        trans_sc = torch.zeros(num_batch, num_res, 3, device=self._device)
        aatypes_sc = torch.zeros(
            num_batch, num_res, self.num_tokens, device=self._device)
        rotvecs_sc = torch.zeros_like(trans_sc)
        torsions_sc = torch.zeros(
            num_batch, num_res, 4, device=self._device
        )
        batch = {
            'res_mask': res_mask,
            'diffuse_mask': diffuse_mask,
            'chain_idx': chain_idx,
            'res_idx': res_idx,
            'torsions_sc': torsions_sc,
            'PLM_emb_weight': PLM_embedder._plm_emb_weight,
        }
        
        logits_1 = torch.nn.functional.one_hot(
            aatypes_1,
            num_classes=self.num_tokens
        ).float()
        if torsions_1 is None:
            torsions_1 = torch.zeros((num_batch, num_res, 4), device=self._device)

        batch['trans_1'] = trans_1
        batch['rotmats_1'] = rotmats_1
        batch['logits_1'] = logits_1
        batch['aatypes_1'] = aatypes_1
        batch['torsions_1'] = torsions_1

        forward_folding = False
        inverse_folding = False
        packing = False
        separate_t = True
        
        batch['aatypes_sc'] = aatypes_sc
        batch['trans_sc'] = trans_sc
        batch['rotvecs_sc'] = rotvecs_sc

        # Set-up time
        if num_timesteps is None:
            num_timesteps = self._sample_cfg.num_timesteps

        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps) # from min_t to 1
        t_1 = ts[0]

        # Set-up the initial input
        trans_t_1, rotmats_t_1, aatypes_t_1, torsions_t_1 = trans_0, rotmats_0, aatypes_0, torsions_0
        
        # Set-up the trajectory
        prot_traj = [(
            all_atom.atom37_from_trans_rot(trans_t_1, rotmats_t_1, res_mask).detach().cpu(), 
            aatypes_0.detach().cpu(),
            all_atom.atom37_from_trans_rot_torsion(trans_t_1, rotmats_t_1, torsions_t_1, aatypes_t_1, res_mask).detach().cpu(),   
        )] # *_traj save the data_t in each time step t
        torsions_traj = [torsions_t_1.detach().cpu()]
        clean_traj = [] # *clean_traj save the data_1 in each time step t
        clean_torsions_traj = []

        # Start sampling
        for t_2 in ts[1:]:

            # Run model.
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            batch['aatypes_t'] = aatypes_t_1
            
            t = torch.ones((num_batch, 1), device=self._device) * t_1
            
            if t_nn is not None:
                batch['r3_t'], batch['so3_t'], batch['cat_t'] = torch.split(t_nn(t), -1)
            else:
                if self._cfg.provide_kappa:
                    batch['so3_t'] = self.rot_sample_kappa(t)
                else:
                    batch['so3_t'] = t
                batch['r3_t'] = t
                batch['cat_t'] = t
                batch['tor_t'] = t 

            d_t = t_2 - t_1

            with torch.no_grad():
                PLM_templates['aatypes_1_design'][PLM_templates['diffuse_mask_design'].bool()] = batch['aatypes_t'][batch['diffuse_mask'].bool()].long()
                if PLM_type == 'faESM2_650M':
                    # faESM2 style
                    PLM_templates['aatypes_1_design'] = torch.where(PLM_templates['res_mask_design'].bool(), PLM_templates['aatypes_1_design'], torch.ones_like(PLM_templates['aatypes_1_design'])*23)
                    max_seqlen = PLM_templates['chain_lengthes_design'].max().item()
                    cu_seqlens = torch.cumsum(PLM_templates['chain_lengthes_design'].view(-1), dim=0)
                    cu_seqlens = torch.cat([torch.tensor([0]).int().to(cu_seqlens.device), cu_seqlens], dim=0)
                    aatypes_in_ESM_templates = PLM_templates['template_design'].view(-1).to(PLM_templates['aatypes_1_design'].dtype)
                    aatypes_in_ESM_templates[~PLM_templates['template_mask_design'].view(-1)] = PLM_templates['aatypes_1_design'].view(-1) # False in template_mask means the non-special tokens
                    plm_s_with_ST = PLM_embedder.faESM2_encoding(aatypes_in_ESM_templates.unsqueeze(0), cu_seqlens.int(), max_seqlen)
                    plm_s_design = plm_s_with_ST[~PLM_templates['template_mask_design'].view(-1)].view(*PLM_templates['aatypes_1_design'].shape, PLM_embedder.plm_representations_layer+1, PLM_embedder.plm_representations_dim)
                    assert not torch.isnan(plm_s_design).any(), "Logits contain NaN values"

                elif PLM_type == 'gLM2_650M':
                    # gLM2 style
                    PLM_templates['aatypes_1_design'] = torch.where(PLM_templates['res_mask_design'].bool(), PLM_templates['aatypes_1_design'], torch.ones_like(PLM_templates['aatypes_1_design'])*22)
                    gLM_template = PLM_templates['template_design'].reshape(-1).to(PLM_templates['aatypes_1_design'].dtype)
                    gLM_template[~PLM_templates['template_mask_design'].reshape(-1)] = PLM_templates['aatypes_1_design'].reshape(-1) # False in template_mask means the non-special tokens
                    gLM_template = gLM_template.reshape(PLM_templates['aatypes_1_design'].shape[0], -1)
                    plm_s_with_ST = PLM_embedder.gLM_encoding(gLM_template)
                    plm_s_design = plm_s_with_ST[~PLM_templates['template_mask_design']].reshape(*PLM_templates['aatypes_1_design'].shape, PLM_embedder.plm_representations_layer+1, PLM_embedder.plm_representations_dim)

                else:
                    raise ValueError(f'Unsupported PLM type {PLM_type}.')

            plm_s_design = plm_s_design.to(batch['trans_t'].dtype)
            PLM_templates['plm_s'][:, :PLM_templates['aatypes_1_design'].shape[1], :, :] = plm_s_design
            batch['PLM_embedding_aatypes_t'] = PLM_templates['plm_s'].reshape(num_batch, -1, *plm_s_design.shape[2:])

            # run the model
            with torch.no_grad():
                backbone_src = 'backbone'
                if model_order == ['backbone', ]:
                    model_out = model['backbone'](batch)
                else:
                    for model_type in model_order:
                        model_out = None
                        if model_type == 'backbone':
                            model_out = model['backbone'](batch)
                        elif model_type == 'sidechain' and t_1 >= self._cfg.sidechain_start_t:
                            model_out = model['sidechain'](batch)
                            tor_pred=True
                        elif model_type == 'refine' and t_1 >= self._cfg.refine_start_t:
                            model_out = model['refine'](batch)
                            backbone_src = 'refine'
                        
                        if not model_out is None:
                            batch = batch | model_out
                    
                    model_out = {k: v for k, v in batch.items() if 'pred' in k}

            # Process model output
            pred_trans_1 = model_out[f'backbone_pred_trans']
            pred_rotmats_1 = model_out[f'backbone_pred_rotmats']
            pred_aatypes_1 = model_out[f'backbone_pred_aatypes']
            pred_logits_1 = model_out[f'backbone_pred_logits']

            sc_trans = pred_trans_1
            sc_rotmats = pred_rotmats_1
            sc_logits = pred_logits_1

            if backbone_src == 'refine':

                refined_trans_1 = model_out[f'refine_pred_trans']
                refined_rotmats_1 = model_out[f'refine_pred_rotmats']
                refined_logits_1 = model_out['refine_pred_logits']
                refined_aatypes_1 = model_out[f'refine_pred_aatypes']

                sc_trans = refined_trans_1
                sc_rotmats = refined_rotmats_1
                sc_logits = refined_logits_1
            
            if tor_pred:
                pred_torsions_1 = model_out['sidechain_pred_torsions']
            else:
                pred_torsions_1 = None

            if self._cfg.self_condition > 0:

                if forward_folding:
                    batch['aatypes_sc'] = logits_1
                    batch['trans_sc'] = _trans_diffuse_mask(
                        sc_trans, trans_1, diffuse_mask)
                    batch['rotvecs_sc'] = _trans_diffuse_mask(
                        so3_utils.rotmat_to_rotvec(sc_rotmats), 
                        so3_utils.rotmat_to_rotvec(rotmats_1),
                        diffuse_mask)
                
                elif inverse_folding:
                    batch['trans_sc'] = trans_1
                    batch['rotvecs_sc'] = so3_utils.rotmat_to_rotvec(rotmats_1)
                    batch['aatypes_sc'] = _trans_diffuse_mask(
                        sc_logits, logits_1, diffuse_mask)
                
                elif packing:
                    batch['aatypes_sc'] = logits_1
                    batch['trans_sc'] = trans_1
                    batch['rotvecs_sc'] = so3_utils.rotmat_to_rotvec(rotmats_1)
                
                else:
                    batch['aatypes_sc'] = _trans_diffuse_mask(
                        sc_logits, logits_1, diffuse_mask)
                    batch['trans_sc'] = _trans_diffuse_mask(
                        sc_trans, trans_1, diffuse_mask)
                    batch['rotvecs_sc'] = _trans_diffuse_mask(
                        so3_utils.rotmat_to_rotvec(sc_rotmats),
                        so3_utils.rotmat_to_rotvec(rotmats_1),
                        diffuse_mask)
                
                # in any situation, groun truth torsion will not be provided
                if tor_pred:
                    # when sidechain is not generated, keep the torsions and torsions_sc as the initial state
                    batch['torsions_sc'] = _torsions_diffuse_mask(pred_torsions_1, torsions_1, diffuse_mask)

            if tor_pred:
                clean_traj.append((
                    all_atom.atom37_from_trans_rot(pred_trans_1, pred_rotmats_1, res_mask).detach().cpu(), 
                    pred_aatypes_1.detach().cpu(),
                    all_atom.atom37_from_trans_rot_torsion(pred_trans_1, pred_rotmats_1, pred_torsions_1, pred_aatypes_1, res_mask).detach().cpu(), 
                ))
                clean_torsions_traj.append(pred_torsions_1.detach().cpu())
            else:
                clean_traj.append((
                    all_atom.atom37_from_trans_rot(pred_trans_1, pred_rotmats_1, res_mask).detach().cpu(),
                    pred_aatypes_1.detach().cpu(),))
            # Take reverse step
            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)
            if rot_style == 'multiflow':            
                rotmats_t_2 = self._rots_euler_step(
                    d_t, t_1, pred_rotmats_1, rotmats_t_1)
            elif rot_style == 'foldflow':
                rot_vectorfield = model_out['rot_vectorfield']
                rotmats_t_2 = foldflow_reverse(rot_t=rotmats_t_1,
                                               v_t=rot_vectorfield,
                                               t=t_1,
                                               dt=d_t,
                                               noise_scale=1.0)
            else:
                raise ValueError(f'{rot_style} has not been implemented')

            if not pred_torsions_1 is None:
                torsions_t_2 = self._torsions_euler_step(d_t, t_1, pred_torsions_1, torsions_t_1)
            else:
                torsions_t_2 = torsions_0

            curr_t = max(1e-4, min(t_2.item(), 1-1e-4))
            if curr_t >= 0.85:
                argmax=True
            else:
                argmax=False

            curr_temp = exp_schedule(curr_t, self._aatypes_cfg.max_temp, self._aatypes_cfg.decay_rate)

            if backbone_src == 'refine':
                aatypes_t_2, aatypes_t_2_scores = self._aatypes_euler_step_APM_Refine(d_t, t_1, pred_logits_1, refined_logits_1, noise_scale=0, temperature=curr_temp, argmax=argmax, diffuse_mask=diffuse_mask)
            else:
                aatypes_t_2, aatypes_t_2_scores = self._aatypes_euler_step_APM(d_t, t_1, pred_logits_1, noise_scale=1.0-curr_t, temperature=curr_temp, argmax=argmax, diffuse_mask=diffuse_mask, apply_penalty=False)

            trans_t_2 = _trans_diffuse_mask(trans_t_2, trans_1, diffuse_mask)
            rotmats_t_2 = _rots_diffuse_mask(rotmats_t_2, rotmats_1, diffuse_mask)
            aatypes_t_2 = _aatypes_diffuse_mask(aatypes_t_2, aatypes_1, diffuse_mask)
            torsions_t_2 = _torsions_diffuse_mask(torsions_t_2, torsions_1, diffuse_mask)
            trans_t_1, rotmats_t_1, aatypes_t_1, torsions_t_1 = trans_t_2, rotmats_t_2, aatypes_t_2, torsions_t_2
            
            if tor_pred:
                prot_traj.append((
                    all_atom.atom37_from_trans_rot(trans_t_1, rotmats_t_1, res_mask).detach().cpu(), 
                    aatypes_0.detach().cpu(),
                    all_atom.atom37_from_trans_rot_torsion(trans_t_1, rotmats_t_1, torsions_t_1, aatypes_t_1, res_mask).detach().cpu(),   
                ))
                torsions_traj.append(torsions_t_2.cpu().detach())
            else:
                prot_traj.append((
                    all_atom.atom37_from_trans_rot(trans_t_1, rotmats_t_1, res_mask).detach().cpu(),
                    aatypes_0.detach().cpu(),))

            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['aatypes_t'] = aatypes_t_1
        
        with torch.no_grad():
            PLM_templates['aatypes_1_design'][PLM_templates['diffuse_mask_design'].bool()] = batch['aatypes_t'][batch['diffuse_mask'].bool()].long()
            if PLM_type == 'faESM2_650M':
                # faESM2 style
                PLM_templates['aatypes_1_design'] = torch.where(PLM_templates['res_mask_design'].bool(), PLM_templates['aatypes_1_design'], torch.ones_like(PLM_templates['aatypes_1_design'])*23)
                max_seqlen = PLM_templates['chain_lengthes_design'].max().item()
                cu_seqlens = torch.cumsum(PLM_templates['chain_lengthes_design'].view(-1), dim=0)
                cu_seqlens = torch.cat([torch.tensor([0]).int().to(cu_seqlens.device), cu_seqlens], dim=0)
                aatypes_in_ESM_templates = PLM_templates['template_design'].view(-1).to(PLM_templates['aatypes_1_design'].dtype)
                aatypes_in_ESM_templates[~PLM_templates['template_mask_design'].view(-1)] = PLM_templates['aatypes_1_design'].view(-1) # False in template_mask means the non-special tokens
                plm_s_with_ST = PLM_embedder.faESM2_encoding(aatypes_in_ESM_templates.unsqueeze(0), cu_seqlens.int(), max_seqlen)
                plm_s_design = plm_s_with_ST[~PLM_templates['template_mask_design'].view(-1)].view(*PLM_templates['aatypes_1_design'].shape, PLM_embedder.plm_representations_layer+1, PLM_embedder.plm_representations_dim)
                assert not torch.isnan(plm_s_design).any(), "Logits contain NaN values"
                
            elif PLM_type == 'gLM2_650M':
                # gLM2 style
                PLM_templates['aatypes_1_design'] = torch.where(PLM_templates['res_mask_design'].bool(), PLM_templates['aatypes_1_design'], torch.ones_like(PLM_templates['aatypes_1_design'])*22)
                gLM_template = PLM_templates['template_design'].reshape(-1).to(PLM_templates['aatypes_1_design'].dtype)
                gLM_template[~PLM_templates['template_mask_design'].reshape(-1)] = PLM_templates['aatypes_1_design'].reshape(-1) # False in template_mask means the non-special tokens
                gLM_template = gLM_template.reshape(PLM_templates['aatypes_1_design'].shape[0], -1)
                plm_s_with_ST = PLM_embedder.gLM_encoding(gLM_template)
                plm_s_design = plm_s_with_ST[~PLM_templates['template_mask_design']].reshape(*PLM_templates['aatypes_1_design'].shape, PLM_embedder.plm_representations_layer+1, PLM_embedder.plm_representations_dim)

            else:
                raise ValueError(f'Unsupported PLM type {PLM_type}.')

        plm_s_design = plm_s_design.to(batch['trans_t'].dtype)
        PLM_templates['plm_s'][:, :PLM_templates['aatypes_1_design'].shape[1], :, :] = plm_s_design
        batch['PLM_embedding_aatypes_t'] = PLM_templates['plm_s'].reshape(num_batch, -1, *plm_s_design.shape[2:])

        with torch.no_grad():
            backbone_src = 'backbone'
            tor_pred = False
            if model_order == ['backbone', ]:
                model_out = model['backbone'](batch)
            else:
                for model_type in model_order:
                    model_out = None
                    if model_type == 'backbone':
                        model_out = model['backbone'](batch)
                    elif model_type == 'sidechain':
                        model_out = model['sidechain'](batch)
                        tor_pred=True
                    elif model_type == 'refine':
                        model_out = model['refine'](batch)
                        backbone_src = 'refine'
                    
                    if not model_out is None:
                        batch = batch | model_out
                
                model_out = {k: v for k, v in batch.items() if 'pred' in k}

        # Process model output.
        backbone_src = 'backbone'

        if backbone_src == 'backbone':
            pred_trans_1 = model_out[f'backbone_pred_trans']
            pred_rotmats_1 = model_out[f'backbone_pred_rotmats']
            pred_aatypes_1 = model_out[f'backbone_pred_aatypes']
            pred_logits_1 = model_out[f'backbone_pred_logits']
        else:
            pred_trans_1 = model_out[f'refine_pred_trans']
            pred_rotmats_1 = model_out[f'refine_pred_rotmats']
            pred_aatypes_1 = model_out[f'refine_pred_aatypes']
            pred_logits_1 = model_out['refine_pred_logits']

        if tor_pred:
            pred_torsions_1 = model_out['sidechain_pred_torsions']
        else:
            pred_torsions_1 = None
            
        pred_trans_1 = _trans_diffuse_mask(pred_trans_1, trans_1, diffuse_mask)
        pred_rotmats_1 = _rots_diffuse_mask(pred_rotmats_1, rotmats_1, diffuse_mask)
        pred_aatypes_1 = _aatypes_diffuse_mask(pred_aatypes_1, aatypes_1, diffuse_mask)
        pred_torsions_1 = _torsions_diffuse_mask(pred_torsions_1, torsions_1, diffuse_mask)

        pred_atom37 = all_atom.atom37_from_trans_rot(pred_trans_1, pred_rotmats_1, res_mask).detach().cpu()
        last_step = [pred_atom37, pred_aatypes_1.detach().cpu(),]
        if tor_pred:
            pred_atom37_Full = all_atom.atom37_from_trans_rot_torsion(pred_trans_1, pred_rotmats_1, pred_torsions_1, pred_aatypes_1, res_mask).detach().cpu()
            last_step.append(pred_atom37_Full)
            clean_torsions_traj.append(pred_torsions_1.detach().cpu())
            torsions_traj.append(pred_torsions_1.detach().cpu())
        
        clean_traj.append(last_step)
        prot_traj.append(last_step)
        
        gt_result = {
            'gt_atom37': all_atom.atom37_from_trans_rot(trans_1, rotmats_1, res_mask).detach().cpu(),
            'gt_aatypes': aatypes_1,
            'gt_atom37_full': all_atom.atom37_from_trans_rot_torsion(trans_1, rotmats_1, torsions_1, aatypes_1, res_mask).detach().cpu(),
        }

        return prot_traj, clean_traj, torsions_traj, clean_torsions_traj, gt_result
