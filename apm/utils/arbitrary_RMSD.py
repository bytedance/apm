"""
Copyright (2025) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0
"""


import os
import torch
import numpy as np
import warnings
from Bio.PDB.PDBParser import PDBParser
from apm.data.residue_constants import residue_atoms, restypes, restype_1to3, atom_types

def load_pdb(pdb_id, pdb_file):
    with warnings.catch_warnings():
        parser = PDBParser(PERMISSIVE=1)
        structure = parser.get_structure(pdb_id, pdb_file)
        protein = structure[0]
    return protein

class rigid_alignment():
    def __init__(self, P_0):

        self.P_0 = P_0
        self.R = torch.zeros(3, 3).to(P_0.device)
        self.T = torch.zeros(1, 3).to(P_0.device)
    
    def set_anchor(self, P_0):
        self.P_0 = P_0
    
    def transform(self, P_1):
        P_1_dim = len(P_1.shape)-1
        T_expand_dim = [1, ] * P_1_dim + [3, ]
        aligned_P_1 = torch.matmul(P_1, self.R) + self.T.reshape(T_expand_dim)
        return aligned_P_1
    
class Kabsch_align(rigid_alignment):

    def align_(self, P_1):

        P_0_center = self.P_0.mean(dim=0) # [3, ]
        P_1_center = P_1.mean(dim=0) # [3, ]

        Q_0 = self.P_0 - P_0_center[None, :] # [N, 3]
        Q_1 = P_1 - P_1_center[None, :] # [N, 3]

        H = torch.matmul(Q_1[:, :, None], Q_0[:, None, :]) # [N, 3, 3]
        H = H.sum(dim=0) # [3, 3]
        U, E, V = torch.linalg.svd(H)
        self.R = torch.matmul(U, V) # [3, 3]
        self.T = self.P_0 - torch.matmul(P_1, self.R)
        self.T = self.T.mean(0) # [3, ]

    def align(self, P_1):

        P_0_center = self.P_0.mean(dim=0) # [3, ]
        P_1_center = P_1.mean(dim=0) # [3, ]

        Q_0 = self.P_0 - P_0_center[None, :] # [N, 3]
        Q_1 = P_1 - P_1_center[None, :] # [N, 3]

        H = torch.matmul(Q_1.T, Q_0) # [3, 3]
        V, S, W = torch.linalg.svd(H)
        d = (torch.linalg.det(V) * torch.linalg.det(W)) < 0.0
        if d:
            V[:, -1] = -V[:, -1]
        self.R = torch.matmul(V, W) # [3, 3]
        
        self.T = self.P_0 - torch.matmul(P_1, self.R)
        self.T = self.T.mean(0) # [3, ]

def cal_RMSD(point_cloud_0, point_cloud_1):
    assert point_cloud_0.shape == point_cloud_1.shape
    squared_distance = torch.sum((point_cloud_0 - point_cloud_1)**2, dim=-1) # [N, ]
    mean_squared_distance = torch.mean(squared_distance)
    root_mean_squared_distance = torch.sqrt(mean_squared_distance)
    return root_mean_squared_distance.cpu().item()

def get_Atom_ids(mode='BB3'):
    mode = mode.upper()
    assert mode in ('CA', 'BB3', 'BB4', 'HEAVY')
    if mode == 'CA':
        valid_ids = ('CA', )
    elif mode == 'BB3':
        valid_ids = ('N', 'CA', 'C')
    elif mode == 'BB4':
        valid_ids = ('N', 'CA', 'C', 'O')
    else:
        valid_ids = ('N', 'CA', 'C', 'CB', 'O', 'CG', 
                     'CG1', 'CG2', 'OG', 'OG1', 'SG', 
                     'CD', 'CD1', 'CD2', 'ND1', 'ND2', 
                     'OD1', 'OD2', 'SD', 'CE', 'CE1', 
                     'CE2', 'CE3', 'NE', 'NE1', 'NE2', 
                     'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 
                     'OH', 'CZ', 'CZ2', 'CZ3', 'NZ', 'OXT')
    return valid_ids

def get_Atom_Coords(pdb_file, mode='BB3'):
    mode = mode.upper()
    assert mode in ('CA', 'BB3', 'BB4', 'HEAVY', 'ALL')
    if mode == 'ALL':
        valid_ids = ()
        constant_valid = True
    else:
        valid_ids = get_Atom_ids(mode=mode)
        constant_valid = False

    pdb_struct = load_pdb('', pdb_file)
    all_chains = list(pdb_struct.child_dict.keys())
    all_chains = sorted(all_chains)
    pdb_coords = {}
    for chain_id in all_chains:
        pdb_coords[chain_id] = []
        chain_struct = pdb_struct.child_dict[chain_id]
        for res in chain_struct.child_list:
            pdb_coords[chain_id].append({})
            for atom in res.child_list:
                atom_id = atom.get_id()
                if atom_id in valid_ids or constant_valid:
                    atom_coords = atom.get_coord()
                    pdb_coords[chain_id][-1][atom_id] = atom_coords
    
    return pdb_coords

def aligned_RMSD(pdb_0, pdb_1, RMSD_on='BB3', aligned_by='CA', device='cpu'):

    # get the RNSD between two pdb files
    # the RMSD_on is the atom type to calculate RMSD
    # the aligned_by is the atom type to align the two pdb files

    RMSD_on = RMSD_on.upper()
    aligned_by = aligned_by.upper()

    assert RMSD_on in ('CA', 'BB3', 'BB4', 'HEAVY')
    assert aligned_by in ('CA', 'BB3', 'BB4', 'HEAVY', '')

    pdb_0_coords = get_Atom_Coords(pdb_0, mode='HEAVY')
    pdb_1_coords = get_Atom_Coords(pdb_1, mode='HEAVY')
    
    pdb_0_chains = list(pdb_0_coords.keys())
    pdb_1_chains = list(pdb_1_coords.keys())

    for chain_id_0, chain_id_1 in zip(pdb_0_chains, pdb_1_chains):
        assert chain_id_0 == chain_id_1
        assert len(pdb_0_coords[chain_id_0]) == len(pdb_1_coords[chain_id_1])

    if aligned_by != '':
        # get the coords used for the alignment
        coords_to_be_aligned_0 = []
        coords_to_be_aligned_1 = []
        aligned_atom_ids = get_Atom_ids(mode=aligned_by)
        for chain_id_0, chain_id_1 in zip(pdb_0_chains, pdb_1_chains):
            for res_0, res_1 in zip(pdb_0_coords[chain_id_0], pdb_1_coords[chain_id_1]):
                res_0_valid_atoms = [atom_id for atom_id in res_0 if atom_id in aligned_atom_ids]
                res_1_valid_atoms = [atom_id for atom_id in res_1 if atom_id in aligned_atom_ids]
                valid_atoms = [atom_id for atom_id in res_0_valid_atoms if atom_id in res_1_valid_atoms]
                assert len(valid_atoms) > 0
                for atom_id in valid_atoms:
                    coords_to_be_aligned_0.append(res_0[atom_id])
                    coords_to_be_aligned_1.append(res_1[atom_id])
        
        coords_to_be_aligned_0 = torch.Tensor(np.array(coords_to_be_aligned_0))
        coords_to_be_aligned_1 = torch.Tensor(np.array(coords_to_be_aligned_1))
        # print('alignment data shape :', coords_to_be_aligned_0.shape, coords_to_be_aligned_1.shape)

    # get the coords used for the RMSD
    coords_to_be_RMSD_0 = []
    coords_to_be_RMSD_1 = []
    RMSD_atom_ids = get_Atom_ids(mode=RMSD_on)
    for chain_id_0, chain_id_1 in zip(pdb_0_chains, pdb_1_chains):
        for res_0, res_1 in zip(pdb_0_coords[chain_id_0], pdb_1_coords[chain_id_1]):
            res_0_valid_atoms = [atom_id for atom_id in res_0 if atom_id in RMSD_atom_ids]
            res_1_valid_atoms = [atom_id for atom_id in res_1 if atom_id in RMSD_atom_ids]
            valid_atoms = [atom_id for atom_id in res_0_valid_atoms if atom_id in res_1_valid_atoms]
            assert len(valid_atoms) > 0
            for atom_id in valid_atoms:
                coords_to_be_RMSD_0.append(res_0[atom_id])
                coords_to_be_RMSD_1.append(res_1[atom_id])
    
    coords_to_be_RMSD_0 = torch.Tensor(np.array(coords_to_be_RMSD_0))
    coords_to_be_RMSD_1 = torch.Tensor(np.array(coords_to_be_RMSD_1))

    if aligned_by != '':
        # prepare data
        if device == 'cuda':
            coords_to_be_aligned_0 = coords_to_be_aligned_0.cuda()
            coords_to_be_aligned_1 = coords_to_be_aligned_1.cuda()
            coords_to_be_RMSD_0 = coords_to_be_RMSD_0.cuda()
            coords_to_be_RMSD_1 = coords_to_be_RMSD_1.cuda()
        
        # alignment
        aligner = Kabsch_align(coords_to_be_aligned_0)
        aligner.align(coords_to_be_aligned_1)
        coords_to_be_RMSD_1_aligned = aligner.transform(coords_to_be_RMSD_1)
    else:
        coords_to_be_RMSD_1_aligned = coords_to_be_RMSD_1

    # calculate RMSD
    # print('RMSD data shape :', coords_to_be_RMSD_0.shape, coords_to_be_RMSD_1_aligned.shape)
    rmsd = cal_RMSD(coords_to_be_RMSD_0, coords_to_be_RMSD_1_aligned)
    
    return rmsd

def build_atom_mask(aatypes):
    if type(aatypes) == str:
        aatypes = [aatypes]
    elif type(aatypes) == np.ndarray:
        aatypes = aatypes.reshape(-1).tolist()
    elif type(aatypes) == torch.Tensor:
        aatypes = aatypes.cpu().reshape(-1).tolist()
    else:
        raise ValueError('aatypes must be str, np.ndarray or torch.Tensor')

    N_res = len(aatypes)
    atom_mask = np.zeros((N_res, 37))
    for res_i, res_type in enumerate(aatypes):
        res_1 = restypes[res_type]
        res_3 = restype_1to3[res_1]
        res_atoms = residue_atoms[res_3]
        for atom_id in res_atoms:
            atom_mask[res_i, atom_types.index(atom_id)] = 1
    
    return atom_mask

def get_target_coords_from_atom37(atom37_coords, mode='BB3', atom_mask=None):
    assert mode in ('CA', 'BB3', 'BB4', 'SC', 'HEAVY', 'O')
    if mode == 'CA':
        target_coords = atom37_coords[:, 1, :] # [N_res, 3]
    elif mode.startswith('BB'):
        target_coords = atom37_coords[:, :3, :] # [N_res, 3, 3]
        if mode == 'BB4':
            target_coords = torch.cat([target_coords, atom37_coords[:, 4:5, :]], dim=1) # [N_res, 4, 3]
            # print(target_coords.shape)
        target_coords = target_coords.reshape(-1, 3) # [N_res * 3/4, 3]
    elif mode == 'O':
        target_coords = atom37_coords[:, 4, :] # [N_res, 1, 3]
    else:
        assert atom_mask is not None, 'atom_mask is required when side chain atoms are required'
        if mode == 'SC':
            atom37_coords = torch.cat([atom37_coords[:, 3:4, :], atom37_coords[:, 5:, :]], dim=1) # [N_res, 33, 3]
            atom_mask = torch.cat([atom_mask[:, 3:4], atom_mask[:, 5:]], dim=1) # [N_res, 33, 3]
        target_coords = atom37_coords[atom_mask.int().long().bool()]
    return target_coords

def aligned_RMSD_coords(pdb_0_coords, pdb_1_coords, RMSD_on='BB3', aligned_by='CA', device='cpu', atom_mask=None):

    # get the RNSD between two pdb coords
    # pdb_0_coords & pdb_1_coords are atom37 format
    # the RMSD_on is the atom type to calculate RMSD
    # the aligned_by is the atom type to align the two pdb

    aligned_by = aligned_by.upper()
    assert aligned_by in ('CA', 'BB3', 'BB4', 'SC', 'HEAVY')

    if type(RMSD_on) == str:
        assert RMSD_on.upper() in ('CA', 'BB3', 'BB4', 'HEAVY', 'O', 'SC')
        RMSD_on = [RMSD_on.upper(), ]
    elif type(RMSD_on) == list or type(RMSD_on) == tuple:
        for atom_type in RMSD_on:
            assert atom_type.upper() in ('CA', 'BB3', 'BB4', 'HEAVY', 'O', 'SC')
        RMSD_on = [atom_type.upper() for atom_type in RMSD_on]
    else:
        raise ValueError('RMSD_on must be a string or a list or a tuple')

    assert pdb_0_coords.shape[0] == pdb_1_coords.shape[0] # [N_res, 37, 3]
    assert pdb_0_coords.shape[-1] == 3
    assert pdb_0_coords.shape[-2] == 37
    assert pdb_1_coords.shape[-1] == 3
    assert pdb_1_coords.shape[-2] == 37

    # get the coords used for the alignment and RMSD calculation
    coords_to_be_aligned_0 = get_target_coords_from_atom37(pdb_0_coords, mode=aligned_by, atom_mask=atom_mask)
    coords_to_be_aligned_1 = get_target_coords_from_atom37(pdb_1_coords, mode=aligned_by, atom_mask=atom_mask)

    # prepare data
    if device == 'cuda':
        coords_to_be_aligned_0 = coords_to_be_aligned_0.cuda()
        coords_to_be_aligned_1 = coords_to_be_aligned_1.cuda()
        coords_to_be_RMSD_0 = coords_to_be_RMSD_0.cuda()
        coords_to_be_RMSD_1 = coords_to_be_RMSD_1.cuda()
    
    # alignment
    aligner = Kabsch_align(coords_to_be_aligned_0)
    aligner.align(coords_to_be_aligned_1)

    rmsd_lib = {}
    for atom_type in RMSD_on:
        coords_to_be_RMSD_0 = get_target_coords_from_atom37(pdb_0_coords, mode=atom_type, atom_mask=atom_mask)
        coords_to_be_RMSD_1 = get_target_coords_from_atom37(pdb_1_coords, mode=atom_type, atom_mask=atom_mask)
        coords_to_be_RMSD_1_aligned = aligner.transform(coords_to_be_RMSD_1)
        rmsd = cal_RMSD(coords_to_be_RMSD_0, coords_to_be_RMSD_1_aligned)
        rmsd_lib[atom_type] = rmsd
    
    return rmsd_lib

def cal_MD(point_cloud_0, point_cloud_1):
    assert point_cloud_0.shape == point_cloud_1.shape
    squared_distance = torch.sum((point_cloud_0 - point_cloud_1)**2, dim=-1) # [N, ]
    distance = torch.sqrt(squared_distance)
    mean_distance = torch.mean(distance)
    return mean_distance.cpu().item()

def aligned_MD_coords(pdb_0_coords, pdb_1_coords, MD_on='BB3', aligned_by='CA', device='cpu', atom_mask=None):

    # MD: Mean Distance

    aligned_by = aligned_by.upper()
    assert aligned_by in ('CA', 'BB3', 'BB4', 'SC', 'HEAVY')

    if type(MD_on) == str:
        assert MD_on.upper() in ('CA', 'BB3', 'BB4', 'HEAVY', 'O', 'SC')
        MD_on = [MD_on.upper(), ]
    elif type(MD_on) == list or type(MD_on) == tuple:
        for atom_type in MD_on:
            assert atom_type.upper() in ('CA', 'BB3', 'BB4', 'HEAVY', 'O', 'SC')
        MD_on = [atom_type.upper() for atom_type in MD_on]
    else:
        raise ValueError('MD_on must be a string or a list or a tuple')

    assert pdb_0_coords.shape[0] == pdb_1_coords.shape[0] # [N_res, 37, 3]
    assert pdb_0_coords.shape[-1] == 3
    assert pdb_0_coords.shape[-2] == 37
    assert pdb_1_coords.shape[-1] == 3
    assert pdb_1_coords.shape[-2] == 37

    # get the coords used for the alignment and RMSD calculation
    coords_to_be_aligned_0 = get_target_coords_from_atom37(pdb_0_coords, mode=aligned_by, atom_mask=atom_mask)
    coords_to_be_aligned_1 = get_target_coords_from_atom37(pdb_1_coords, mode=aligned_by, atom_mask=atom_mask)

    # prepare data
    if device == 'cuda':
        coords_to_be_aligned_0 = coords_to_be_aligned_0.cuda()
        coords_to_be_aligned_1 = coords_to_be_aligned_1.cuda()
        coords_to_be_RMSD_0 = coords_to_be_RMSD_0.cuda()
        coords_to_be_RMSD_1 = coords_to_be_RMSD_1.cuda()
    
    # alignment
    aligner = Kabsch_align(coords_to_be_aligned_0)
    aligner.align(coords_to_be_aligned_1)

    md_lib = {}
    for atom_type in MD_on:
        coords_to_be_RMSD_0 = get_target_coords_from_atom37(pdb_0_coords, mode=atom_type, atom_mask=atom_mask)
        coords_to_be_RMSD_1 = get_target_coords_from_atom37(pdb_1_coords, mode=atom_type, atom_mask=atom_mask)
        coords_to_be_RMSD_1_aligned = aligner.transform(coords_to_be_RMSD_1)
        md = cal_MD(coords_to_be_RMSD_0, coords_to_be_RMSD_1_aligned)
        md_lib[atom_type] = md
    
    return md_lib

if __name__ == '__main__':
    pdb_0 = 'length_100/sample_0/sample_full.pdb'
    pdb_1 = 'length_100/sample_0/self_consistency/folded/folded_codesign_seq_1.pdb'
    rmsd = aligned_RMSD(pdb_0, pdb_1, RMSD_on='Heavy', aligned_by='CA', device='cpu')
    print(rmsd)


    gen_path = 'length_500/samples/'
    fold_path = 'length_500/folded/'
    scRMSD = {}
    all_ids = [i.split('_')[1][:4] for i in os.listdir(gen_path)]
    for i in all_ids:
        gen_pdb = f'{gen_path}sample_{i}.pdb'
        folded_pdb = f'{fold_path}folded_{i}.pdb'
        sr = aligned_RMSD(gen_pdb, folded_pdb, RMSD_on='CA', aligned_by='CA', device='cpu')
        scRMSD[i] = sr