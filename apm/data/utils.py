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


import dataclasses
import numpy as np
import collections
import string
import pickle
import os
import torch
from typing import List, Dict, Any
from openfold.utils import rigid_utils as ru
from torch_scatter import scatter_add, scatter
from Bio.PDB.Chain import Chain
from Bio import PDB
from apm.data import protein, residue_constants, parsers
from glob import glob
from pytorch_lightning.utilities import rank_zero_only
import random

Rigid = ru.Rigid
Protein = protein.Protein

# Global map from chain characters to integers.
ALPHANUMERIC = string.ascii_letters + string.digits + ' '
CHAIN_TO_INT = {
    chain_char: i for i, chain_char in enumerate(ALPHANUMERIC)
}
INT_TO_CHAIN = {
    i: chain_char for i, chain_char in enumerate(ALPHANUMERIC)
}

NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE

CHAIN_FEATS = [
    'atom_positions', 'aatype', 'atom_mask', 'residue_index', 'b_factors'
]

NUM_TOKENS = residue_constants.restype_num
MASK_TOKEN_INDEX = residue_constants.restypes_with_x.index('X')
CA_IDX = residue_constants.atom_order['CA']

to_numpy = lambda x: x.detach().cpu().numpy()
aatype_to_seq = lambda aatype: ''.join([
        residue_constants.restypes_with_x[x] for x in aatype])
seq_to_aatype = lambda seq: [
        residue_constants.restypes_with_x.index(x) for x in seq]


class CPU_Unpickler(pickle.Unpickler):
    """Pytorch pickle loading workaround.

    https://github.com/pytorch/pytorch/issues/16797
    """
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def remove_com_from_tensor_7(tensor_7):
    """remove_com_from_tensor_7 removes the center of mass along the L dimension from a tensor of shape (..., L, 7).
    """
    rigid = Rigid.from_tensor_7(tensor_7)
    rigid_trans = rigid.get_trans()
    COM = rigid_trans.mean(dim=[-2], keepdim=True)
    rigid_trans = rigid_trans - COM
    rigid._trans = rigid_trans
    return rigid.to_tensor_7()


def create_rigid(rots, trans):
    rots = ru.Rotation(rot_mats=rots)
    return Rigid(rots=rots, trans=trans)


def batch_align_structures(pos_1, pos_2, mask=None):
    if pos_1.shape != pos_2.shape:
        raise ValueError('pos_1 and pos_2 must have the same shape.')
    if pos_1.ndim != 3:
        raise ValueError(f'Expected inputs to have shape [B, N, 3]')
    num_batch = pos_1.shape[0]
    device = pos_1.device
    batch_indices = (
        torch.ones(*pos_1.shape[:2], device=device, dtype=torch.int64) 
        * torch.arange(num_batch, device=device)[:, None]
    ) # [B, N] : [[0,0,0,...,0],[1,1,1,....,1],[2,2,2,...,2],[3,3,3,...,3],...]
    flat_pos_1 = pos_1.reshape(-1, 3) # [B*L, 3]
    flat_pos_2 = pos_2.reshape(-1, 3) # [B*L, 3]
    flat_batch_indices = batch_indices.reshape(-1) # [B*L]
    if mask is None:
        aligned_pos_1, aligned_pos_2, align_rots = align_structures(
            flat_pos_1, flat_batch_indices, flat_pos_2)
        aligned_pos_1 = aligned_pos_1.reshape(num_batch, -1, 3)
        aligned_pos_2 = aligned_pos_2.reshape(num_batch, -1, 3)
        return aligned_pos_1, aligned_pos_2, align_rots

    flat_mask = mask.reshape(-1).bool()
    _, _, align_rots = align_structures(
        flat_pos_1[flat_mask],
        flat_batch_indices[flat_mask],
        flat_pos_2[flat_mask]
    )
    aligned_pos_1 = torch.bmm(
        pos_1,
        align_rots
    )
    return aligned_pos_1, pos_2, align_rots


def adjust_oxygen_pos(
    atom_37: torch.Tensor, pos_is_known = None
) -> torch.Tensor:
    """
    Imputes the position of the oxygen atom on the backbone by using adjacent frame information.
    Specifically, we say that the oxygen atom is in the plane created by the Calpha and C from the
    current frame and the nitrogen of the next frame. The oxygen is then placed c_o_bond_length Angstrom
    away from the C in the current frame in the direction away from the Ca-C-N triangle.

    For cases where the next frame is not available, for example we are at the C-terminus or the
    next frame is not available in the data then we place the oxygen in the same plane as the
    N-Ca-C of the current frame and pointing in the same direction as the average of the
    Ca->C and Ca->N vectors.

    Args:
        atom_37 (torch.Tensor): (N, 37, 3) tensor of positions of the backbone atoms in atom_37 ordering
                                which is ['N', 'CA', 'C', 'CB', 'O', ...]
        pos_is_known (torch.Tensor): (N,) mask for known residues.
    """

    N = atom_37.shape[0]
    assert atom_37.shape == (N, 37, 3)

    # Get vectors to Carbonly from Carbon alpha and N of next residue. (N-1, 3)
    # Note that the (N,) ordering is from N-terminal to C-terminal.

    # Calpha to carbonyl both in the current frame.
    calpha_to_carbonyl: torch.Tensor = (atom_37[:-1, 2, :] - atom_37[:-1, 1, :]) / (
        torch.norm(atom_37[:-1, 2, :] - atom_37[:-1, 1, :], keepdim=True, dim=1) + 1e-7
    )
    # For masked positions, they are all 0 and so we add 1e-7 to avoid division by 0.
    # The positions are in Angstroms and so are on the order ~1 so 1e-7 is an insignificant change.

    # Nitrogen of the next frame to carbonyl of the current frame.
    nitrogen_to_carbonyl: torch.Tensor = (atom_37[:-1, 2, :] - atom_37[1:, 0, :]) / (
        torch.norm(atom_37[:-1, 2, :] - atom_37[1:, 0, :], keepdim=True, dim=1) + 1e-7
    )

    carbonyl_to_oxygen: torch.Tensor = calpha_to_carbonyl + nitrogen_to_carbonyl  # (N-1, 3)
    carbonyl_to_oxygen = carbonyl_to_oxygen / (
        torch.norm(carbonyl_to_oxygen, dim=1, keepdim=True) + 1e-7
    )

    atom_37[:-1, 4, :] = atom_37[:-1, 2, :] + carbonyl_to_oxygen * 1.23

    # Now we deal with frames for which there is no next frame available.

    # Calpha to carbonyl both in the current frame. (N, 3)
    calpha_to_carbonyl_term: torch.Tensor = (atom_37[:, 2, :] - atom_37[:, 1, :]) / (
        torch.norm(atom_37[:, 2, :] - atom_37[:, 1, :], keepdim=True, dim=1) + 1e-7
    )
    # Calpha to nitrogen both in the current frame. (N, 3)
    calpha_to_nitrogen_term: torch.Tensor = (atom_37[:, 0, :] - atom_37[:, 1, :]) / (
        torch.norm(atom_37[:, 0, :] - atom_37[:, 1, :], keepdim=True, dim=1) + 1e-7
    )
    carbonyl_to_oxygen_term: torch.Tensor = (
        calpha_to_carbonyl_term + calpha_to_nitrogen_term
    )  # (N, 3)
    carbonyl_to_oxygen_term = carbonyl_to_oxygen_term / (
        torch.norm(carbonyl_to_oxygen_term, dim=1, keepdim=True) + 1e-7
    )

    # Create a mask that is 1 when the next residue is not available either
    # due to this frame being the C-terminus or the next residue is not
    # known due to pos_is_known being false.

    if pos_is_known is None:
        pos_is_known = torch.ones((atom_37.shape[0],), dtype=torch.int64, device=atom_37.device)

    next_res_gone: torch.Tensor = ~pos_is_known.bool()  # (N,)
    next_res_gone = torch.cat(
        [next_res_gone, torch.ones((1,), device=pos_is_known.device).bool()], dim=0
    )  # (N+1, )
    next_res_gone = next_res_gone[1:]  # (N,)

    atom_37[next_res_gone, 4, :] = (
        atom_37[next_res_gone, 2, :]
        + carbonyl_to_oxygen_term[next_res_gone, :] * 1.23
    )

    return atom_37


def write_pkl(
        save_path: str, pkl_data: Any, create_dir: bool = False, use_torch=False):
    """Serialize data into a pickle file."""
    if create_dir:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if use_torch:
        torch.save(pkl_data, save_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(save_path, 'wb') as handle:
            pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pkl(read_path: str, verbose=True, use_torch=False, map_location=None):
    """Read data from a pickle file."""
    try:
        if use_torch:
            return torch.load(read_path, map_location=map_location)
        else:
            with open(read_path, 'rb') as handle:
                return pickle.load(handle)
    except Exception as e:
        try:
            with open(read_path, 'rb') as handle:
                return CPU_Unpickler(handle).load()
        except Exception as e2:
            if verbose:
                print(f'Failed to read {read_path}. First error: {e}\n Second error: {e2}')
            raise(e)

def chain_str_to_int(chain_str: str):
    chain_int = 0
    if len(chain_str) == 1:
        return CHAIN_TO_INT[chain_str]
    for i, chain_char in enumerate(chain_str):
        chain_int += CHAIN_TO_INT[chain_char] + (i * len(ALPHANUMERIC))
    return chain_int

def parse_chain_feats(chain_feats, scale_factor=1., center=True):
    chain_feats['bb_mask'] = chain_feats['atom_mask'][:, CA_IDX]
    bb_pos = chain_feats['atom_positions'][:, CA_IDX]
    if center:
        bb_center = np.sum(bb_pos, axis=0) / (np.sum(chain_feats['bb_mask']) + 1e-5)
        centered_pos = chain_feats['atom_positions'] - bb_center[None, None, :]
        scaled_pos = centered_pos / scale_factor
    else:
        scaled_pos = chain_feats['atom_positions'] / scale_factor
    chain_feats['atom_positions'] = scaled_pos * chain_feats['atom_mask'][..., None]
    chain_feats['bb_positions'] = chain_feats['atom_positions'][:, CA_IDX]
    return chain_feats

def concat_np_features(
        np_dicts: List[Dict[str, np.ndarray]], add_batch_dim: bool):
    """Performs a nested concatenation of feature dicts.

    Args:
        np_dicts: list of dicts with the same structure.
            Each dict must have the same keys and numpy arrays as the values.
        add_batch_dim: whether to add a batch dimension to each feature.

    Returns:
        A single dict with all the features concatenated.
    """
    combined_dict = collections.defaultdict(list)
    for chain_dict in np_dicts:
        for feat_name, feat_val in chain_dict.items():
            if add_batch_dim:
                feat_val = feat_val[None]
            combined_dict[feat_name].append(feat_val)
    # Concatenate each feature
    for feat_name, feat_vals in combined_dict.items():
        combined_dict[feat_name] = np.concatenate(feat_vals, axis=0)
    return combined_dict


def center_zero(pos: torch.Tensor, batch_indexes: torch.LongTensor) -> torch.Tensor:
    """
    Move the molecule center to zero for sparse position tensors.

    Args:
        pos: [N, 3] batch positions of atoms in the molecule in sparse batch format.
        batch_indexes: [N] batch index for each atom in sparse batch format.

    Returns:
        pos: [N, 3] zero-centered batch positions of atoms in the molecule in sparse batch format.
    """
    assert len(pos.shape) == 2 and pos.shape[-1] == 3, "pos must have shape [N, 3]"

    means = scatter(pos, batch_indexes, dim=0, reduce="mean")
    return pos - means[batch_indexes]


@torch.no_grad()
def align_structures(
    batch_positions: torch.Tensor,
    batch_indices: torch.Tensor,
    reference_positions: torch.Tensor,
    broadcast_reference: bool = False,
):
    """
    Align structures in a ChemGraph batch to a reference, e.g. for RMSD computation. This uses the
    sparse formulation of pytorch geometric. If the ChemGraph is composed of a single system, then
    the reference can be given as a single structure and broadcasted. Returns the structure
    coordinates shifted to the geometric center and the batch structures rotated to match the
    reference structures. Uses the Kabsch algorithm (see e.g. [kabsch_align1]_). No permutation of
    atoms is carried out.

    Args:
        batch_positions (Tensor): Batch of structures (e.g. from ChemGraph) which should be aligned
          to a reference.
        batch_indices (Tensor): Index tensor mapping each node / atom in batch to the respective
          system (e.g. batch attribute of ChemGraph batch).
        reference_positions (Tensor): Reference structure. Can either be a batch of structures or a
          single structure. In the second case, broadcasting is possible if the input batch is
          composed exclusively of this structure.
        broadcast_reference (bool, optional): If reference batch contains only a single structure,
          broadcast this structure to match the ChemGraph batch. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors containing the centered positions of batch
          structures rotated into the reference and the centered reference batch.

    References
    ----------
    .. [kabsch_align1] Lawrence, Bernal, Witzgall:
       A purely algebraic justification of the Kabsch-Umeyama algorithm.
       Journal of research of the National Institute of Standards and Technology, 124, 1. 2019.
    """
    # Minimize || Q @ R.T - P ||, which is the same as || Q - P @ R ||
    # batch_positions     -> P [BN x 3]
    # reference_positions -> Q [B / BN x 3]

    if batch_positions.shape[0] != reference_positions.shape[0]:
        if broadcast_reference:
            # Get number of systems in batch and broadcast reference structure.
            # This assumes, all systems in the current batch correspond to the reference system.
            # Typically always the case during evaluation.
            num_molecules = int(torch.max(batch_indices) + 1)
            reference_positions = reference_positions.repeat(num_molecules, 1)
        else:
            raise ValueError("Mismatch in batch dimensions.")

    # Center structures at origin (takes care of translation alignment)
    batch_positions = center_zero(batch_positions, batch_indices)
    reference_positions = center_zero(reference_positions, batch_indices)

    # Compute covariance matrix for optimal rotation (Q.T @ P) -> [B x 3 x 3].
    cov = scatter_add(
        batch_positions[:, None, :] * reference_positions[:, :, None], batch_indices, dim=0
    )

    # Perform singular value decomposition. (all [B x 3 x 3])
    u, _, v_t = torch.linalg.svd(cov)
    # Convenience transposes.
    u_t = u.transpose(1, 2)
    v = v_t.transpose(1, 2)

    # Compute rotation matrix correction for ensuring right-handed coordinate system
    # For comparison with other sources: det(AB) = det(A)*det(B) and det(A) = det(A.T)
    sign_correction = torch.sign(torch.linalg.det(torch.bmm(v, u_t).float()))
    # Correct transpose of U: diag(1, 1, sign_correction) @ U.T
    u_t[:, 2, :] = u_t[:, 2, :] * sign_correction[:, None]

    # Compute optimal rotation matrix (R = V @ diag(1, 1, sign_correction) @ U.T).
    rotation_matrices = torch.bmm(v, u_t)

    # Rotate batch positions P to optimal alignment with Q (P @ R)
    batch_positions_rotated = torch.bmm(
        batch_positions[:, None, :],
        rotation_matrices[batch_indices],
    ).squeeze(1)

    return batch_positions_rotated, reference_positions, rotation_matrices


def parse_pdb_feats(
        pdb_name: str,
        pdb_path: str,
        scale_factor=1.,
        # TODO: Make the default behaviour read all chains.
        chain_id='A',
    ):
    """
    Args:
        pdb_name: name of PDB to parse.
        pdb_path: path to PDB file to read.
        scale_factor: factor to scale atom positions.
        mean_center: whether to mean center atom positions.
    Returns:
        Dict with CHAIN_FEATS features extracted from PDB with specified
        preprocessing.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_name, pdb_path)
    struct_chains = {
        chain.id: chain
        for chain in structure.get_chains()}

    def _process_chain_id(x):
        chain_prot = process_chain(struct_chains[x], x)
        chain_dict = dataclasses.asdict(chain_prot)

        # Process features
        feat_dict = {x: chain_dict[x] for x in CHAIN_FEATS}
        return parse_chain_feats(
            feat_dict, scale_factor=scale_factor)

    if isinstance(chain_id, str):
        return _process_chain_id(chain_id)
    elif isinstance(chain_id, list):
        return {
            x: _process_chain_id(x) for x in chain_id
        }
    elif chain_id is None:
        return {
            x: _process_chain_id(x) for x in struct_chains
        }
    else:
        raise ValueError(f'Unrecognized chain list {chain_id}')


def process_chain(chain: Chain, chain_id: str) -> Protein:
    """Convert a PDB chain object into a AlphaFold Protein instance.

    Forked from alphafold.common.protein.from_pdb_string

    WARNING: All non-standard residue types will be converted into UNK. All
        non-standard atoms will be ignored.

    Took out lines 94-97 which don't allow insertions in the PDB.
    Sabdab uses insertions for the chothia numbering so we need to allow them.

    Took out lines 110-112 since that would mess up CDR numbering.

    Args:
        chain: Instance of Biopython's chain class.

    Returns:
        Protein object with protein features.
    """
    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    b_factors = []
    chain_ids = []
    for res in chain:
        res_shortname = residue_constants.restype_3to1.get(res.resname, 'X')
        restype_idx = residue_constants.restype_order.get(
            res_shortname, residue_constants.restype_num)
        pos = np.zeros((residue_constants.atom_type_num, 3))
        mask = np.zeros((residue_constants.atom_type_num,))
        res_b_factors = np.zeros((residue_constants.atom_type_num,))
        for atom in res:
            if atom.name not in residue_constants.atom_types:
                continue
            pos[residue_constants.atom_order[atom.name]] = atom.coord
            mask[residue_constants.atom_order[atom.name]] = 1.
            res_b_factors[residue_constants.atom_order[atom.name]
                          ] = atom.bfactor
        aatype.append(restype_idx)
        atom_positions.append(pos)
        atom_mask.append(mask)
        residue_index.append(res.id[1])
        b_factors.append(res_b_factors)
        chain_ids.append(chain_id)

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=np.array(chain_ids),
        b_factors=np.array(b_factors))


def extract_sequence_from_pdb(file_path):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('decoy', file_path)

    # Extract all chains
    struct_chains = {
        chain.id.upper(): chain
        for chain in structure.get_chains()}
    if len(struct_chains) > 1:
        raise ValueError('Only compatible with monomers')

    # Convert chain id into int
    chain_id = list(struct_chains.keys())[0]
    chain = struct_chains[chain_id]
    chain_prot = parsers.process_chain(chain, chain_id)
    chain_dict = dataclasses.asdict(chain_prot)
    chain_dict = parse_chain_feats(chain_dict)
    return chain_dict['aatype']


def get_synthetic_data_folder(dataset_cfg):
    synthetic_data_folders = glob(os.path.join(os.path.dirname(dataset_cfg.synthetic_data_folder), "*"))
    target_pid = os.getpid() if rank_zero_only.rank == 0 else os.getppid()
    for folder in synthetic_data_folders:
        current_date = os.path.dirname(dataset_cfg.synthetic_data_folder).split('_')[0]
        if current_date in folder and str(target_pid) in folder:
            return folder
    print("Didnt find any matching synthetic data folder")
    return None


def randint(lower, upper, device):
    return int(torch.randint(
        lower,
        upper + 1,
        (1,),
        device=device,
    )[0])


def get_interface_residues(positions, atom_mask, asym_id, interface_threshold):
    coord_diff = positions[..., None, :, :] - positions[..., None, :, :, :]
    pairwise_dists = torch.sqrt(torch.sum(coord_diff ** 2, dim=-1))

    diff_chain_mask = (asym_id[..., None, :] != asym_id[..., :, None]).float()
    pair_mask = atom_mask[..., None, :] * atom_mask[..., None, :, :]
    mask = (diff_chain_mask[..., None] * pair_mask).bool()

    min_dist_per_res, _ = torch.where(mask, pairwise_dists, torch.inf).min(dim=-1)

    valid_interfaces = torch.sum((min_dist_per_res < interface_threshold).float(), dim=-1)
    interface_residues_idxs = torch.nonzero(valid_interfaces, as_tuple=True)[0]

    return interface_residues_idxs


def get_spatial_crop_idx(protein, crop_size, interface_threshold=5.0):
    positions = protein["all_atom_positions"]
    atom_mask = protein["all_atom_mask"]
    asym_id = protein["asym_id"]

    interface_residues = get_interface_residues(positions=positions,
                                                atom_mask=atom_mask,
                                                asym_id=asym_id,
                                                interface_threshold=interface_threshold)
    
    if not torch.any(interface_residues):
        # print(f"Detected no interface residues. Using all residues instead")
        interface_residues = torch.arange(0, positions.shape[0], device=positions.device)

    target_res_idx = randint(lower=0,
                             upper=interface_residues.shape[-1] - 1,
                             device=positions.device)

    target_res = interface_residues[target_res_idx]

    ca_idx = residue_constants.atom_order["CA"]
    ca_positions = positions[..., ca_idx, :]
    ca_mask = atom_mask[..., ca_idx].bool()

    coord_diff = ca_positions[..., None, :] - ca_positions[..., None, :, :]
    ca_pairwise_dists = torch.sqrt(torch.sum(coord_diff ** 2, dim=-1))

    to_target_distances = ca_pairwise_dists[target_res]
    break_tie = (
            torch.arange(
                0, to_target_distances.shape[-1], device=positions.device
            ).float()
            * 1e-3
    )
    to_target_distances = torch.where(ca_mask, to_target_distances, torch.inf) + break_tie

    ret = torch.argsort(to_target_distances)[:crop_size]
    return ret.sort().values

def get_spatial_crop_idx_simplfy(protein, crop_size, interface_threshold=12.0):
    CA_positions = protein["all_atom_positions"][:, 1, :].float()
    CB_positions = protein["all_atom_positions"][:, 3, :].float()
    CB_atom_mask = protein["all_atom_mask"][:, 3,].float()
    ATOM_positions = CB_positions * CB_atom_mask[...,None] + CA_positions * (1 - CB_atom_mask[...,None])
    RES_mask = protein["all_atom_mask"][:, 1].float()
    asym_id = protein["asym_id"]

    coord_diff = ATOM_positions[:, None, :] - ATOM_positions[None, :, :] # [L, L, 3]
    squared_pairwise_dists = torch.sum(coord_diff ** 2, dim=-1) # [L, L]

    pair_mask = RES_mask[:, None] * RES_mask[None, :] # [L, L]
    squared_pairwise_dists += pair_mask * 1e8 # distance involves invalid res is set to a large value

    squared_interface_threshold = interface_threshold ** 2

    diff_chain_mask = asym_id[None, :].eq(asym_id[:, None]).float() # [L, L]
    inter_chain_squared_dists = squared_pairwise_dists + diff_chain_mask * 1e8
    valid_interfaces = torch.sum(inter_chain_squared_dists.lt(squared_interface_threshold), dim=-1) # [L]
    interface_masks = valid_interfaces.gt(0).long() # [L]

    N_total_res = protein["all_atom_positions"].shape[0]
    if N_total_res <= crop_size:
        return torch.ones(N_total_res), interface_masks, None

    interface_residues_idxs = torch.nonzero(valid_interfaces, as_tuple=True)[0]

    if not torch.any(interface_residues_idxs):
        # print(f"Detected no interface residues. Using all residues instead")
        interface_residues_idxs = torch.arange(CA_positions.shape[0])

    target_res = random.choice(interface_residues_idxs.tolist())
    to_target_res_squared_dist = squared_pairwise_dists[target_res]
    crop_indices = to_target_res_squared_dist.argsort()[:crop_size].sort().values
    crop_masks = torch.zeros(CA_positions.shape[0])
    crop_masks[crop_indices] = 1
    return crop_masks, interface_masks, target_res

def find_binding_surface_residues(ca_coords, cutoff_distance=6.0, top_rank=0.5, ignore=0.25):
    """
    Finds potential binding surface residues using a scoring method.

    Args:
        ca_coords (torch.Tensor): CA atom coordinates of shape [N, 3].
        cutoff_distance (float): Distance threshold for neighbor detection.
        top_rank (float): Proportion of top-scoring residues to select.
        ignore (float): Proportion of top residues to ignore.

    Returns:
        int: Index of a randomly selected surface residue.
    """
    N = ca_coords.shape[0]
    
    # 1. Compute protein centroid
    centroid = torch.mean(ca_coords, dim=0)
    
    # 2. Compute distances from each CA to centroid
    to_centroid = ca_coords - centroid
    distances_to_centroid = torch.norm(to_centroid, dim=1)
    
    # 3. Count neighbors within cutoff distance
    distances = torch.cdist(ca_coords, ca_coords)
    neighbor_counts = torch.sum((distances < cutoff_distance) & (distances > 0), dim=1)
    
    # 4. Calculate surface scores (far from centroid + fewer neighbors)
    normalized_distances = distances_to_centroid / torch.max(distances_to_centroid)
    normalized_neighbors = 1 - (neighbor_counts / torch.max(neighbor_counts))
    surface_scores = normalized_distances + normalized_neighbors
    
    # Select top-ranked residues, ignoring a portion
    k = max(1, int(N * top_rank))
    _, top_indices = torch.topk(surface_scores, k)
    top_indices = top_indices[int(len(top_indices) * ignore):]
    
    # Return a random residue from the selected set
    random_idx = np.random.randint(0, len(top_indices))
    return top_indices[random_idx].item()

def create_sequence_templates(seq_lengths, bos_id=21, eos_id=22):
    """
    Create two types of sequence templates
    Args:
        seq_lengths: list of sequence lengths
        bos_id: ID for beginning of sequence token
        eos_id: ID for end of sequence token
    Returns:
        Dictionary with two tensors and their masks
    """
    # Remove padding zeros
    seq_lengths = [length for length in seq_lengths if length != 0]
    
    # Calculate correct lengths
    # Pattern1: each sequence needs 1 BOS + sequence_length
    pattern1_length = len(seq_lengths) + sum(seq_lengths)
    
    # Pattern2: each sequence needs 1 BOS + sequence_length + 1 EOS
    pattern2_length = sum([length + 2 for length in seq_lengths])
    
    # Initialize tensors with correct lengths
    pattern1 = torch.zeros(pattern1_length, dtype=torch.long)
    pattern2 = torch.zeros(pattern2_length, dtype=torch.long)
    
    # Fill pattern1: <bos>s1<bos>s2...
    pos1 = 0
    for length in seq_lengths:
        pattern1[pos1] = bos_id
        pos1 += length + 1
    
    # Fill pattern2: <bos>s1<eos><bos>s2<eos>...
    curr_pos = 0
    for length in seq_lengths:
        pattern2[curr_pos] = bos_id  # Add BOS
        curr_pos += length + 1  # Move to EOS position
        pattern2[curr_pos] = eos_id  # Add EOS
        curr_pos += 1  # Move to next sequence start
    
    # Create masks
    pattern1_mask = pattern1.ne(0)
    pattern2_mask = pattern2.ne(0)

    return {
        "gLM_template": pattern1,
        "gLM_template_mask": pattern1_mask,
        "faESM2_template": pattern2,
        "faESM2_template_mask": pattern2_mask
    }
