"""
----------------
Copyright 2021 AlQuraishi Laboratory
Copyright 2021 DeepMind Technologies Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
----------------
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”). 
All Bytedance's Modifications are Copyright (2025) Bytedance Ltd. and/or its affiliates. 
"""


import numpy as np
import torch
from openfold.np import residue_constants as RC
from openfold.utils.tensor_utils import batched_gather
from openfold.utils.rigid_utils import Rotation, Rigid
from openfold.utils.geometry.rigid_matrix_vector import Rigid3Array
from openfold.utils.geometry.rotation_matrix import Rot3Array
from openfold.utils.geometry.vector import Vec3Array
from openfold.data.data_transforms import get_chi_atom_indices




def atom37_to_backbone_frame(protein, eps=1e-8):
    is_multimer = "asym_id" in protein
    aatype = protein["aatype"] # [N_res,]
    all_atom_positions = protein["all_atom_positions"].double() # [N_res, 37, 3]
    all_atom_mask = protein["all_atom_mask"] # [N_res, 37]

    assert not is_multimer, "This protein is a multimer." 
    # zhouxiangxin: this assert is for only modeling single chain.
    if is_multimer:
        all_atom_positions = Vec3Array.from_array(all_atom_positions)

    batch_dims = len(aatype.shape[:-1])

    restype_rigidgroup_base_atom_names = np.full([21, 8, 3], "", dtype=object) # [21, 8, 3]
    restype_rigidgroup_base_atom_names[:, 0, :] = ["C", "CA", "N"]
    restype_rigidgroup_base_atom_names[:, 3, :] = ["CA", "C", "O"]

    for restype, restype_letter in enumerate(RC.restypes):
        resname = RC.restype_1to3[restype_letter]
        for chi_idx in range(4):
            if RC.chi_angles_mask[restype][chi_idx]:
                names = RC.chi_angles_atoms[resname][chi_idx]
                restype_rigidgroup_base_atom_names[
                    restype, chi_idx + 4, :
                ] = names[1:]

    restype_rigidgroup_mask = all_atom_mask.new_zeros(
        (*aatype.shape[:-1], 21, 8),
    ) # [21, 8]
    restype_rigidgroup_mask[..., 0] = 1
    restype_rigidgroup_mask[..., 3] = 1
    restype_rigidgroup_mask[..., :20, 4:] = all_atom_mask.new_tensor(
        RC.chi_angles_mask
    )

    lookuptable = RC.atom_order.copy()
    lookuptable[""] = 0
    lookup = np.vectorize(lambda x: lookuptable[x])
    restype_rigidgroup_base_atom37_idx = lookup(
        restype_rigidgroup_base_atom_names,
    )
    restype_rigidgroup_base_atom37_idx = aatype.new_tensor(
        restype_rigidgroup_base_atom37_idx,
    )
    restype_rigidgroup_base_atom37_idx = (
        restype_rigidgroup_base_atom37_idx.view(
            *((1,) * batch_dims), *restype_rigidgroup_base_atom37_idx.shape
        )
    )

    residx_rigidgroup_base_atom37_idx = batched_gather(
        restype_rigidgroup_base_atom37_idx,
        aatype,
        dim=-3,
        no_batch_dims=batch_dims,
    )

    if is_multimer:
        base_atom_pos = [batched_gather(
            pos,
            residx_rigidgroup_base_atom37_idx,
            dim=-1,
            no_batch_dims=len(all_atom_positions.shape[:-1]),
        ) for pos in all_atom_positions]
        base_atom_pos = Vec3Array.from_array(torch.stack(base_atom_pos, dim=-1))
    else:
        base_atom_pos = batched_gather(
            all_atom_positions,
            residx_rigidgroup_base_atom37_idx,
            dim=-2,
            no_batch_dims=len(all_atom_positions.shape[:-2]),
        )

    if is_multimer:
        point_on_neg_x_axis = base_atom_pos[:, :, 0]
        origin = base_atom_pos[:, :, 1]
        point_on_xy_plane = base_atom_pos[:, :, 2]
        gt_rotation = Rot3Array.from_two_vectors(
            origin - point_on_neg_x_axis, point_on_xy_plane - origin)

        gt_frames = Rigid3Array(gt_rotation, origin)
    else:
        gt_frames = Rigid.from_3_points(
            p_neg_x_axis=base_atom_pos[..., 0, :],
            origin=base_atom_pos[..., 1, :],
            p_xy_plane=base_atom_pos[..., 2, :],
            eps=eps,
        )

    group_exists = batched_gather(
        restype_rigidgroup_mask,
        aatype,
        dim=-2,
        no_batch_dims=batch_dims,
    )

    gt_atoms_exist = batched_gather(
        all_atom_mask,
        residx_rigidgroup_base_atom37_idx,
        dim=-1,
        no_batch_dims=len(all_atom_mask.shape[:-1]),
    )
    gt_exists = torch.min(gt_atoms_exist, dim=-1)[0] * group_exists

    rots = torch.eye(3, dtype=all_atom_mask.dtype, device=aatype.device)
    rots = torch.tile(rots, (*((1,) * batch_dims), 8, 1, 1))
    rots[..., 0, 0, 0] = -1
    rots[..., 0, 2, 2] = -1

    if is_multimer:
        gt_frames = gt_frames.compose_rotation(
            Rot3Array.from_array(rots))
    else:
        rots = Rotation(rot_mats=rots)
        gt_frames = gt_frames.compose(Rigid(rots, None))

    restype_rigidgroup_is_ambiguous = all_atom_mask.new_zeros(
        *((1,) * batch_dims), 21, 8
    )
    restype_rigidgroup_rots = torch.eye(
        3, dtype=all_atom_mask.dtype, device=aatype.device
    )
    restype_rigidgroup_rots = torch.tile(
        restype_rigidgroup_rots,
        (*((1,) * batch_dims), 21, 8, 1, 1),
    )

    for resname, _ in RC.residue_atom_renaming_swaps.items():
        restype = RC.restype_order[RC.restype_3to1[resname]]
        chi_idx = int(sum(RC.chi_angles_mask[restype]) - 1)
        restype_rigidgroup_is_ambiguous[..., restype, chi_idx + 4] = 1
        restype_rigidgroup_rots[..., restype, chi_idx + 4, 1, 1] = -1
        restype_rigidgroup_rots[..., restype, chi_idx + 4, 2, 2] = -1

    # residx_rigidgroup_is_ambiguous = batched_gather(
    #     restype_rigidgroup_is_ambiguous,
    #     aatype,
    #     dim=-2,
    #     no_batch_dims=batch_dims,
    # )

    residx_rigidgroup_ambiguity_rot = batched_gather(
        restype_rigidgroup_rots,
        aatype,
        dim=-4,
        no_batch_dims=batch_dims,
    )

    if is_multimer:
        pass
        # ambiguity_rot = Rot3Array.from_array(residx_rigidgroup_ambiguity_rot)

        # # Create the alternative ground truth frames.
        # alt_gt_frames = gt_frames.compose_rotation(ambiguity_rot)
    else:
        residx_rigidgroup_ambiguity_rot = Rotation(
            rot_mats=residx_rigidgroup_ambiguity_rot
        )
        # alt_gt_frames = gt_frames.compose(
        #     Rigid(residx_rigidgroup_ambiguity_rot, None)
        # )

    gt_frames_tensor = gt_frames.to_tensor_4x4()
    # alt_gt_frames_tensor = alt_gt_frames.to_tensor_4x4()

    # protein["rigidgroups_gt_frames"] = gt_frames_tensor
    # protein["rigidgroups_gt_exists"] = gt_exists
    # protein["rigidgroups_group_exists"] = group_exists
    # protein["rigidgroups_group_is_ambiguous"] = residx_rigidgroup_is_ambiguous
    # protein["rigidgroups_alt_gt_frames"] = alt_gt_frames_tensor

    # protein["backbone_rigid_orientation"] = gt_frames_tensor[
    #     ..., 0, :3, :3
    # ] # [N_res, 3, 3]
    backbone_rigid_orientation = gt_frames_tensor[..., 0, :3, :3] # [N_res, 3, 3]
    backbone_rigid_mask = gt_exists[..., 0] # [N_res, ]

    return gt_frames_tensor, backbone_rigid_orientation, backbone_rigid_mask



def get_chi_atom_indices():
    """Returns atom indices needed to compute chi angles for all residue types.

    Returns:
      A tensor of shape [residue_types=21, chis=4, atoms=4]. The residue types are
      in the order specified in rc.restypes + unknown residue type
      at the end. For chi angles which are not defined on the residue, the
      positions indices are by default set to 0.
    """
    chi_atom_indices = []
    for residue_name in RC.restypes:
        residue_name = RC.restype_1to3[residue_name]
        residue_chi_angles = RC.chi_angles_atoms[residue_name]
        atom_indices = []
        for chi_angle in residue_chi_angles:
            atom_indices.append([RC.atom_order[atom] for atom in chi_angle])
        for _ in range(4 - len(atom_indices)):
            atom_indices.append(
                [0, 0, 0, 0]
            )  # For chi angles not defined on the AA.
        chi_atom_indices.append(atom_indices)

    chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.

    # [21, 4, 4], dim0: residue type, dim1: chi index, dim2: atom index

    return chi_atom_indices

def atom37_to_torsion_angles(protein):

    # calculation of 3 backbone torsion angles (omega, phi, psi) and 4 sidechain torsion angles (chi1 - chi4)
    # from openfold

    """
    Convert coordinates to torsion angles.

    This function is extremely sensitive to floating point imprecisions
    and should be run with double precision whenever possible.

    Args:
        Dict containing:
            * (prefix)aatype:
                [*, N_res] residue indices
            * (prefix)all_atom_positions:
                [*, N_res, 37, 3] atom positions (in atom37
                format)
            * (prefix)all_atom_mask:
                [*, N_res, 37] atom position mask
    Returns:
        The same dictionary updated with the following features:

        "(prefix)torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Torsion angles
        "(prefix)alt_torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Alternate torsion angles (accounting for 180-degree symmetry)
        "(prefix)torsion_angles_mask" ([*, N_res, 7])
            Torsion angles mask
    """
    aatype = protein["aatype"] # [N_res]
    all_atom_positions = protein["all_atom_positions"].double() # [N_res, 37, 3]
    all_atom_mask = protein["all_atom_mask"] # [N_res, 37]

    aatype = torch.clamp(aatype, max=20)

    pad = all_atom_positions.new_zeros(
        [*all_atom_positions.shape[:-3], 1, 37, 3] # [1, 37, 3]
    )
    prev_all_atom_positions = torch.cat(
        [pad, all_atom_positions[..., :-1, :, :]], dim=-3
    ) # [1, 37, 3] + [N_res-1, 37, 3] -> [N_res, 37, 3] dim0:[pad, Res_0, Res_1, ..., Res_-2]

    pad = all_atom_mask.new_zeros([*all_atom_mask.shape[:-2], 1, 37]) # [*, 1, 37]
    prev_all_atom_mask = torch.cat([pad, all_atom_mask[..., :-1, :]], dim=-2) # [*, 1, 37] + [*, N_res-1, 37] -> [*, N_res, 37]

    pre_omega_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 1:3, :], all_atom_positions[..., :2, :]],
        dim=-2,
    ) # [N_res, [prev_CA, prev_C], 3] + [N_res, [curr_N, curr_CA], 3] -> [N_res, [prev_CA, prev_C, curr_N, curr_CA], 3]
    phi_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 2:3, :], all_atom_positions[..., :3, :]],
        dim=-2,
    ) # [N_res, [prev_C, ], 3] + [N_res, [curr_N, curr_CA, curr_C], 3] -> [N_res, [prev_C, curr_N, curr_CA, curr_C], 3]
    psi_atom_pos = torch.cat(
        [all_atom_positions[..., :3, :], all_atom_positions[..., 4:5, :]],
        dim=-2,
    ) # [N_res, [curr_N, curr_CA, curr_C], 3] + [N_res, [curr_O], 3] -> [N_res, [curr_N, curr_CA, curr_C, curr_O], 3]
    # psi is calculated by N-CA-C-O instead of N-CA-C-N

    pre_omega_mask = torch.prod(
        prev_all_atom_mask[..., 1:3], dim=-1
    ) * torch.prod(all_atom_mask[..., :2], dim=-1) 
    # [N_res] * [N_res] -> [N_res] : the mask indicates whether omega exists

    phi_mask = prev_all_atom_mask[..., 2] * torch.prod(
        all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype
    )
    # [N_res] * [N_res] -> [N_res] : the mask indicates whether phi exists
    psi_mask = (
        torch.prod(all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype)
        * all_atom_mask[..., 4]
    )
    # [N_res] * [N_res] -> [N_res] : the mask indicates whether psi exists
    chi_atom_indices = torch.as_tensor(
        get_chi_atom_indices(), device=aatype.device
    )
    # [21, 4, 4]

    atom_indices = chi_atom_indices[..., aatype, :, :] # [N_res, 4, 4]
    chis_atom_pos = batched_gather(
        all_atom_positions, atom_indices, -2, len(atom_indices.shape[:-2])
    ) # [N_res, 4, 4, 3]

    chi_angles_mask = list(RC.chi_angles_mask)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = all_atom_mask.new_tensor(chi_angles_mask)

    chis_mask = chi_angles_mask[aatype, :]

    chi_angle_atoms_mask = batched_gather(
        all_atom_mask,
        atom_indices,
        dim=-1,
        no_batch_dims=len(atom_indices.shape[:-2]),
    )
    chi_angle_atoms_mask = torch.prod(
        chi_angle_atoms_mask, dim=-1, dtype=chi_angle_atoms_mask.dtype
    )
    chis_mask = chis_mask * chi_angle_atoms_mask

    torsions_atom_pos = torch.cat(
        [
            pre_omega_atom_pos[..., None, :, :],
            phi_atom_pos[..., None, :, :],
            psi_atom_pos[..., None, :, :],
            chis_atom_pos,
        ],
        dim=-3,
    )

    torsion_angles_mask = torch.cat(
        [
            pre_omega_mask[..., None],
            phi_mask[..., None],
            psi_mask[..., None],
            chis_mask,
        ],
        dim=-1,
    )

    torsion_frames = Rigid.from_3_points(
        torsions_atom_pos[..., 1, :],
        torsions_atom_pos[..., 2, :],
        torsions_atom_pos[..., 0, :],
        eps=1e-8,
    )

    fourth_atom_rel_pos = torsion_frames.invert().apply(
        torsions_atom_pos[..., 3, :]
    )

    torsion_angles_sin_cos = torch.stack(
        [fourth_atom_rel_pos[..., 2], fourth_atom_rel_pos[..., 1]], dim=-1
    )

    denom = torch.sqrt(
        torch.sum(
            torch.square(torsion_angles_sin_cos),
            dim=-1,
            dtype=torsion_angles_sin_cos.dtype,
            keepdims=True,
        )
        + 1e-8
    )
    torsion_angles_sin_cos = torsion_angles_sin_cos / denom

    torsion_angles_sin_cos = torsion_angles_sin_cos * all_atom_mask.new_tensor(
        [1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
    )[((None,) * len(torsion_angles_sin_cos.shape[:-2])) + (slice(None), None)]

    # chi_is_ambiguous = torsion_angles_sin_cos.new_tensor(
    #     RC.chi_pi_periodic,
    # )[aatype, ...]

    # mirror_torsion_angles = torch.cat(
    #     [
    #         all_atom_mask.new_ones(*aatype.shape, 3),
    #         1.0 - 2.0 * chi_is_ambiguous,
    #     ],
    #     dim=-1,
    # )

    # alt_torsion_angles_sin_cos = (
    #     torsion_angles_sin_cos * mirror_torsion_angles[..., None]
    # )

    # protein["torsion_angles_sin_cos"] = torsion_angles_sin_cos
    # protein["alt_torsion_angles_sin_cos"] = alt_torsion_angles_sin_cos
    # protein["torsion_angles_mask"] = torsion_angles_mask

    # torsion_angles_sin_cos [N_res, 7, 2]
    torsion_angles = torch.Tensor([torch.arctan2(res_sin_cos[:, 0], res_sin_cos[:, 1]).tolist() for res_sin_cos in torsion_angles_sin_cos])
    #torsion_angles = torch.Tensor(torsion_angles)

    return torsion_angles_sin_cos, torsion_angles, torsion_angles_mask


def arc_sin_cos(sin, cos, output_unit="pi"):
    theta_ = torch.acos(cos)
    theta_p_sin = torch.sin(theta_)
    theta_n_sin = torch.sin(-1*theta_)
    
    p_gap = abs(theta_p_sin-sin)
    n_gap = abs(theta_n_sin-sin)
    
    if p_gap < n_gap:
        theta = theta_
    else:
        theta = -1*theta_
    
    if output_unit=='pi':
        return theta
    else:
        return 180*theta/torch.pi
