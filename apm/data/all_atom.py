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


"""Utilities for calculating all atom representations."""
import torch
from openfold.data import data_transforms
from openfold.np import residue_constants
from openfold.utils import rigid_utils as ru
from apm.data import utils as du
from openfold.np import residue_constants as rc
from openfold.utils import tensor_utils as tu
from openfold.utils.feats import frames_and_literature_positions_to_atom14_pos

Rigid = ru.Rigid
Rotation = ru.Rotation

# Residue Constants from OpenFold/AlphaFold2.


IDEALIZED_POS = torch.tensor(residue_constants.restype_atom14_rigid_group_positions)
DEFAULT_FRAMES = torch.tensor(residue_constants.restype_rigid_group_default_frame)
ATOM_MASK = torch.tensor(residue_constants.restype_atom14_mask)
GROUP_IDX = torch.tensor(residue_constants.restype_atom14_to_rigid_group)


def to_atom37(trans, rots):
    num_batch, num_res, _ = trans.shape
    final_atom37 = compute_backbone(
        du.create_rigid(rots, trans),
        torch.zeros(num_batch, num_res, 2, device=trans.device)
    )[0]
    return final_atom37


def torsion_angles_to_frames(
    r: Rigid,  # type: ignore [valid-type]
    alpha: torch.Tensor,
    aatype: torch.Tensor,
):
    """Conversion method of torsion angles to frames provided the backbone.

    Args:
        r: Backbone rigid groups.
        alpha: Torsion angles. [B, N_res, 7, 2], first 3 are zeros, last 4 are sidechian torsions, 2 are cos and sin.
        aatype: residue types.

    Returns:
        All 8 frames corresponding to each torsion frame.

    """
    # [*, N, 8, 4, 4]
    with torch.no_grad():
        default_4x4 = DEFAULT_FRAMES.to(aatype.device)[aatype, ...]  # type: ignore [attr-defined]

    # [*, N, 8] transformations, i.e.
    #   One [*, N, 8, 3, 3] rotation matrix and
    #   One [*, N, 8, 3]    translation matrix
    default_r = r.from_tensor_4x4(default_4x4)  # type: ignore [attr-defined]

    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 1] = 1

    # [*, N, 8, 2]
    alpha = torch.cat([bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2)

    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.

    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha

    all_rots = Rigid(Rotation(rot_mats=all_rots), None)

    all_frames = default_r.compose(all_rots)

    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)  # type: ignore [index]

    return all_frames_to_global


def prot_to_torsion_angles(aatype, atom37, atom37_mask):
    """Calculate torsion angle features from protein features."""
    prot_feats = {
        "aatype": aatype,
        "all_atom_positions": atom37,
        "all_atom_mask": atom37_mask,
    }
    torsion_angles_feats = data_transforms.atom37_to_torsion_angles()(prot_feats)
    torsion_angles = torsion_angles_feats["torsion_angles_sin_cos"]
    torsion_mask = torsion_angles_feats["torsion_angles_mask"]
    return torsion_angles, torsion_mask


def frames_to_atom14_pos(
    r: Rigid,  # type: ignore [valid-type]
    aatype: torch.Tensor,
):
    """Convert frames to their idealized all atom representation.

    Args:
        r: All rigid groups. [..., N, 8, 3]
        aatype: Residue types. [..., N]

    Returns:

    """
    with torch.no_grad():
        group_mask = GROUP_IDX.to(aatype.device)[aatype, ...]
        group_mask = torch.nn.functional.one_hot(
            group_mask,
            num_classes=DEFAULT_FRAMES.shape[-3],
        )
        frame_atom_mask = ATOM_MASK.to(aatype.device)[aatype, ...].unsqueeze(-1)  # type: ignore [attr-defined]
        frame_null_pos = IDEALIZED_POS.to(aatype.device)[aatype, ...]  # type: ignore [attr-defined]

    # [*, N, 14, 8]
    t_atoms_to_global = r[..., None, :] * group_mask  # type: ignore [index]

    # [*, N, 14]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(lambda x: torch.sum(x, dim=-1))

    # [*, N, 14, 3]
    pred_positions = t_atoms_to_global.apply(frame_null_pos)
    pred_positions = pred_positions * frame_atom_mask

    return pred_positions


def compute_backbone(bb_rigids, psi_torsions):
    torsion_angles = torch.tile(
        psi_torsions[..., None, :], tuple([1 for _ in range(len(bb_rigids.shape))]) + (7, 1)
    )
    aatype = torch.zeros(bb_rigids.shape, device=bb_rigids.device).long()
    # aatype = torch.zeros(bb_rigids.shape).long().to(bb_rigids.device)
    all_frames = torsion_angles_to_frames(
        bb_rigids,
        torsion_angles,
        aatype,
    )
    atom14_pos = frames_to_atom14_pos(all_frames, aatype)
    atom37_bb_pos = torch.zeros(bb_rigids.shape + (37, 3), device=bb_rigids.device)
    # atom14 bb order = ['N', 'CA', 'C', 'O', 'CB']
    # atom37 bb order = ['N', 'CA', 'C', 'CB', 'O']
    atom37_bb_pos[..., :3, :] = atom14_pos[..., :3, :]
    atom37_bb_pos[..., 3, :] = atom14_pos[..., 4, :]
    atom37_bb_pos[..., 4, :] = atom14_pos[..., 3, :]
    atom37_mask = torch.any(atom37_bb_pos, axis=-1)
    return atom37_bb_pos, atom37_mask, aatype, atom14_pos


def calculate_neighbor_angles(R_ac, R_ab):
    """Calculate angles between atoms c <- a -> b.

    Parameters
    ----------
        R_ac: Tensor, shape = (N,3)
            Vector from atom a to c.
        R_ab: Tensor, shape = (N,3)
            Vector from atom a to b.

    Returns
    -------
        angle_cab: Tensor, shape = (N,)
            Angle between atoms c <- a -> b.
    """
    # cos(alpha) = (u * v) / (|u|*|v|)
    x = torch.sum(R_ac * R_ab, dim=1)  # shape = (N,)
    # sin(alpha) = |u x v| / (|u|*|v|)
    y = torch.cross(R_ac, R_ab).norm(dim=-1)  # shape = (N,)
    # avoid that for y == (0,0,0) the gradient wrt. y becomes NaN
    y = torch.max(y, torch.tensor(1e-9))
    angle = torch.atan2(y, x)
    return angle


def vector_projection(R_ab, P_n):
    """
    Project the vector R_ab onto a plane with normal vector P_n.

    Parameters
    ----------
        R_ab: Tensor, shape = (N,3)
            Vector from atom a to b.
        P_n: Tensor, shape = (N,3)
            Normal vector of a plane onto which to project R_ab.

    Returns
    -------
        R_ab_proj: Tensor, shape = (N,3)
            Projected vector (orthogonal to P_n).
    """
    a_x_b = torch.sum(R_ab * P_n, dim=-1)
    b_x_b = torch.sum(P_n * P_n, dim=-1)
    return R_ab - (a_x_b / b_x_b)[:, None] * P_n


def transrot_traj_to_atom37(transrot_traj, res_mask):
    atom37_traj = []
    for trans, rots in transrot_traj:
        atom37_traj.append(atom37_from_trans_rot(trans, rots, res_mask))
    return atom37_traj


def atom37_from_trans_rot(trans, rots, res_mask = None):
    if res_mask is None:
        res_mask = torch.ones([*trans.shape[:-1]], device=trans.device)
    rigids = du.create_rigid(rots, trans)
    atom37 = compute_backbone(
        rigids,
        torch.zeros(
            trans.shape[0],
            trans.shape[1],
            2,
            device=trans.device
        )
    )[0]
    batch_atom37 = []
    num_batch = res_mask.shape[0]
    for i in range(num_batch):
        batch_atom37.append(
            du.adjust_oxygen_pos(atom37[i], res_mask[i])
        )
    return torch.stack(batch_atom37)

def create_denser_atom_position():
    """Construct denser atom positions (14 dimensions instead of 37)."""
    # restype_atom14_to_atom37 = []
    restype_atom37_to_atom14 = []
    # restype_atom14_mask = []

    for rt in rc.restypes:
        atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
        # restype_atom14_to_atom37.append(
        #     [(rc.atom_order[name] if name else 0) for name in atom_names]
        # )
        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append(
            [
                (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
                for name in rc.atom_types
            ]
        )

    # Add dummy mapping for restype 'UNK'
    # restype_atom14_to_atom37.append([0] * 14)
    restype_atom37_to_atom14.append([0] * 37)
    # restype_atom14_mask.append([0.0] * 14)
    restype_atom37_to_atom14 = torch.tensor(
        restype_atom37_to_atom14,
        dtype=torch.int32,
    )

    restype_atom37_mask = torch.zeros(
        [21, 37], dtype=torch.float32
    )

    for restype, restype_letter in enumerate(rc.restypes):
        restype_name = rc.restype_1to3[restype_letter]
        atom_names = rc.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = rc.atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1
    return restype_atom37_to_atom14, restype_atom37_mask



def atom37_from_trans_rot_torsion(trans, rots, torsion, aatype, res_mask=None, purify_mask=True):
    aatype = aatype.long()
    
    aatype[aatype == du.MASK_TOKEN_INDEX] = 0

    if res_mask is None:
        res_mask = torch.ones([*trans.shape[:-1]], device=trans.device)
    rigids = du.create_rigid(rots, trans)
    assert torsion.ndim == 3
    batch_size, res_len, angle_num = torsion.shape
    assert angle_num == 4
    torsion_sin_cos = torch.cat((torch.sin(torsion[..., None]), torch.cos(torsion[..., None])), dim=-1) # [B, N_res, 4, 2]
    torsion_full_sin_cos = torch.zeros(size=(batch_size, res_len, 3, 2), device=torsion.device)
    torsion_full_sin_cos = torch.cat([torsion_full_sin_cos, torsion_sin_cos], dim=2) # [B, N_res, 7, 2], first 3 are zeros, last 4 are sidechian torsions


    all_frames = torsion_angles_to_frames(
        rigids,
        torsion_full_sin_cos,
        aatype,
    )
    
    atom14 = frames_and_literature_positions_to_atom14_pos(
        all_frames,
        aatype,
        DEFAULT_FRAMES.to(aatype.device),
        GROUP_IDX.to(aatype.device),
        ATOM_MASK.to(aatype.device),
        IDEALIZED_POS.to(aatype.device),
    )
    
    restype_atom37_to_atom14, restype_atom37_mask = create_denser_atom_position()
    
    residx_atom37_mask = restype_atom37_mask.to(atom14.device)[aatype]
    residx_atom37_to_atom14 = restype_atom37_to_atom14.to(atom14.device)[aatype]
        

    atom37 = tu.batched_gather(
        atom14,
        residx_atom37_to_atom14.long(),
        dim=-2,
        no_batch_dims=len(atom14.shape[:-2]),
    )
    atom37 = atom37 * residx_atom37_mask[..., None]


    batch_atom37 = []
    num_batch = res_mask.shape[0]
    for i in range(num_batch):
        batch_atom37.append(
            du.adjust_oxygen_pos(atom37[i], res_mask[i])
        )
    return torch.stack(batch_atom37)
