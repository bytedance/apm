"""
Copyright (2025) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0
"""


import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import random
import os
import time

from glob import glob
from torch.utils.data import Dataset
from apm.data import utils as du
from openfold.data import data_transforms
from openfold.utils import rigid_utils
from openfold.np import residue_constants
from apm.data.full_atom_utils import atom37_to_torsion_angles
from apm.data.interpolant import Interpolant, _centered_gaussian, _uniform_so3, _masked_categorical, _uniform_torsion

from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Polypeptide import is_aa


valid_residue_types = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS', 
                       'MET','PHE','PRO','SER','THR','TRP','TYR','VAL','SEC','PYL','UNK',]


class StandardAASelect(Select):
    """Selects only standard amino acids."""
    def accept_residue(self, residue):
        # Skip HETATM or water molecules
        if residue.id[0] != " ":
            return 0
        # Keep standard amino acids
        return is_aa(residue)


def clean_pdb(input_pdb, output_pdb):
    """Cleans a PDB file by keeping only standard amino acids."""
    if os.path.exists(output_pdb):
        print(f"{output_pdb} already exists.")
        return
    try:
        parser = PDBParser(QUIET=True)  # Suppress warnings
        structure = parser.get_structure('protein', input_pdb)
        io = PDBIO()
        io.set_structure(structure)
        io.save(output_pdb, StandardAASelect())  # Save cleaned structure
        print(f"Cleaning complete! Saved to {output_pdb}")
    except Exception as e:
        print(f"Error: {str(e)}")


def read_structure(file_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', file_path)
    model = structure[0] 
    return model


def get_info_from_Bio(Bio_structure, target_chain_ids=None):
    if target_chain_ids is None:
        target_chains = [chain for chain in Bio_structure]
    else:
        target_chains = [Bio_structure[c_id] for c_id in target_chain_ids]
    
    info = {}

    for chain in target_chains:
        chain_id = chain.get_id()
        chain_seq = ''
        chain_coords = []
        chain_mask = []
        chain_atoms = []
        chain_bfactors = []
        chain_res_seq = []
        chain_res_icode = []
        chain_valid = True
        UNK_res_mask = []
        HET_res_mask = []

        valid_residue_index = []

        for residue_index, residue in enumerate(chain):

            residue_type = residue.resname

            if residue_type == 'MSE':
                residue_type = 'MET'

            het, resseq, icode = residue.get_id()

            if residue_type in valid_residue_types:
                # only record amino acid

                if residue_type in ('SEC','PYL','UNK'):
                    UNK_res_mask.append(1)
                    valid_atom_ids = ["N", "CA", "C", "O"]
                else:
                    UNK_res_mask.append(0)
                    valid_atom_ids = residue_constants.restype_name_to_atom14_names[residue_type]
                
                if het == ' ':
                    HET_res_mask.append(0)
                else:
                    HET_res_mask.append(1)

                chain_seq += residue_constants.restype_3to1.get(residue_type, "X") # 'SEC','PYL','UNK' are both treated as "X"
                chain_res_seq.append(resseq)
                chain_res_icode.append(icode)
                valid_residue_index.append(residue_index)
                residue_atoms = []
                for atom in residue:
                    atom_id = atom.get_name().strip()
                    if atom_id.upper() == 'SE':
                        atom_id = 'SD'
                    if atom_id in valid_atom_ids:
                        chain_coords.append(atom.get_coord())
                        chain_mask.append(residue_index)
                        residue_atoms.append(atom_id)
                        chain_bfactors.append(atom.get_bfactor())
                chain_atoms.append(residue_atoms)

            elif residue_type in nucleic_acid:
                # when nucleic acid exists, it is not a protein chain
                chain_valid = False
                break

            else:
                pass
        
        index_trans_table = {}
        for actual_residue_index, mask_residue_index in enumerate(valid_residue_index):
            index_trans_table[mask_residue_index] = actual_residue_index
        chain_mask_correct = [index_trans_table[idx] for idx in chain_mask]

        chain_coords = np.array(chain_coords)

        # if chain_valid and len(chain_seq)>50:
        info[chain_id] = {'seq':chain_seq, 
                            'flatten_coords':chain_coords, 
                            'coords_mask':chain_mask_correct, 
                            'atom_ids':chain_atoms, 
                            'b_factors':chain_bfactors, 
                            'chain_res_seq':chain_res_seq, 
                            'chain_res_icode':chain_res_icode, 
                            'chain_UNK_mask':UNK_res_mask, 
                            'chain_HET_mask':HET_res_mask}

    return info


def chain_meta_info_to_AF_style(chain_meta_info):

    meta_seq = chain_meta_info['seq']
    meta_coords = chain_meta_info['flatten_coords']
    meta_mask = chain_meta_info['coords_mask']
    meta_atom_ids = chain_meta_info['atom_ids']
    meta_atom_ids = sum(meta_atom_ids, [])
    meta_b_factors = chain_meta_info['b_factors']
    meta_res_seq = chain_meta_info['chain_res_seq']
    meta_res_icode = chain_meta_info['chain_res_icode']
    meta_UNK_mask = chain_meta_info['chain_UNK_mask']
    meta_HET_mask = chain_meta_info['chain_HET_mask']

    atom_positions = []
    atom_positions_14 = []
    atom14_to_atom37 = []
    aatype = []
    atom_mask = []
    atom_mask_14 = []
    b_factors = []

    # residue level information
    for res_type in meta_seq:
        restype_idx = residue_constants.restype_order.get(res_type, residue_constants.restype_num)
        aatype.append(restype_idx)

    # atom level information
    last_residue_index = -1
    for curr_residue_index, atom_id, atom_coord, bf in zip(meta_mask, meta_atom_ids, meta_coords, meta_b_factors):

        if curr_residue_index != last_residue_index:
            atom_positions.append(np.zeros((37, 3)))
            atom_positions_14.append([])
            atom14_to_atom37.append([])
            atom_mask.append(np.zeros((37, )))
            atom_mask_14.append([])
            b_factors.append(np.zeros((37, )))
            last_residue_index = curr_residue_index

        atom_positions[-1][residue_constants.atom_order[atom_id]] = atom_coord
        atom_positions_14[-1].append(atom_coord)
        atom14_to_atom37[-1].append(residue_constants.atom_order[atom_id])
        atom_mask[-1][residue_constants.atom_order[atom_id]] = 1.0
        atom_mask_14[-1].append(1)
        b_factors[-1][residue_constants.atom_order[atom_id]] = bf
    
    for residue_index in range(len(atom14_to_atom37)):
        padding_length = 14 - len(atom14_to_atom37[residue_index])
        atom_positions_14[residue_index] += [[0.0, 0.0, 0.0] for _ in range(padding_length)]
        # if len(atom_positions_14[residue_index]) != 14:
        #     print(residue_index)
        atom14_to_atom37[residue_index] += [0 for _ in range(padding_length)]
        atom_mask_14[residue_index] += [0 for _ in range(padding_length)]

    chain_AF_stype = {'aatype' : torch.Tensor(np.array(aatype)).long(), # [N_res,]
                      'aaindex' : torch.Tensor(list(range(len(aatype)))), # [N_res,]
                      'res_seq' : torch.Tensor(meta_res_seq).long(), # [N_res,]
                      'res_UNK' : torch.Tensor(meta_UNK_mask).long(), # [N_res,]
                      'res_HET' : torch.Tensor(meta_HET_mask).long(), # [N_res,]
                      'atom_positions' : torch.Tensor(np.array(atom_positions)), # [N_res, 37, 3]
                      'atom_positions_14' : torch.Tensor(np.array(atom_positions_14)), # [N_res, 14, 3]
                      'atom14_to_atom37' : torch.Tensor(np.array(atom14_to_atom37)).long(), # [N_res, 14]
                      'atom_mask' : torch.Tensor(np.array(atom_mask)), # [N_res, 37]
                      'atom_mask_14' : torch.Tensor(np.array(atom_mask_14)).long(), # [N_res, 14]
                      'b_factors' : torch.Tensor(np.array(b_factors)), # [N_res, 37]
                      }
    
    return chain_AF_stype


def _process_csv_row(processed_feats):

    # Run through OpenFold data transforms.
    chain_feats = {
        'aatype': processed_feats['aatype'].long(),
        'all_atom_positions': processed_feats['atom_positions'].double(),
        'all_atom_mask': processed_feats['atom_mask'].double()
    }
    chain_feats = data_transforms.atom37_to_frames(chain_feats)

    torsion_angles_sin_cos, torsion_angles, torsion_angles_mask = \
        atom37_to_torsion_angles(chain_feats)
    chain_feats['torsion_angles'] = torsion_angles[:, -4:]

    rigids_1 = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]
    rotmats_1 = rigids_1.get_rots().get_rot_mats()
    trans_1 = rigids_1.get_trans()
    res_mask = processed_feats['bb_mask'].int()

    # Re-number residue indices for each chain such that it starts from 1.
    # Randomize chain indices.
    chain_idx = processed_feats['chain_index'].numpy()
    res_idx = processed_feats['residue_index'].numpy()
    new_res_idx = np.zeros_like(res_idx)
    new_chain_idx = np.zeros_like(res_idx)
    all_chain_idx = np.unique(chain_idx).tolist()
    shuffled_chain_idx = np.array([i+1 for i in range(len(all_chain_idx))])
    for i,chain_id in enumerate(all_chain_idx):
        chain_mask = (chain_idx == chain_id).astype(int)
        chain_min_idx = np.min(res_idx + (1 - chain_mask) * 1e3).astype(int)
        new_res_idx = new_res_idx + (res_idx - chain_min_idx + 1) * chain_mask

        # Shuffle chain_index
        replacement_chain_id = shuffled_chain_idx[i]
        new_chain_idx = new_chain_idx + replacement_chain_id * chain_mask

    return_dict =  {
        'aatypes_1': chain_feats['aatype'],
        'rotmats_1': rotmats_1,
        'trans_1': trans_1,
        'torsions_1': chain_feats['torsion_angles'],
        # 'torsions_mask': torsion_angles_mask[:, -4:], 
        'res_mask': torch.ones_like(res_mask),
        'chain_idx': torch.tensor(new_chain_idx),
        'res_idx': torch.tensor(new_res_idx),
        'diffuse_mask': processed_feats['diffuse_mask'],
        # 'all_atom_positions': torch.tensor(chain_feats['all_atom_positions']),
    }
    return return_dict



class ConditionalDataset(Dataset):
    def __init__(self, dataset_cfg, plm_style='faESM2'):
        self._dataset_cfg = dataset_cfg
        self.pdb_raw = dataset_cfg.pdb_path
        self.pdb_clean = dataset_cfg.pdb_path.replace('.pdb', '_clean.pdb')
        self.pdb_name = os.path.basename(dataset_cfg.pdb_path).split('.')[0]
        self.sample_num = dataset_cfg.sample_num
        self.sample_length = dataset_cfg.sample_length
        self.chain_design = dataset_cfg.chain_design
        self.random_coefficient = dataset_cfg.random_coefficient

        if plm_style.startswith('gLM'):
            self._plm_style = 'gLM2'
        elif plm_style.startswith('faESM') :
            self._plm_style = 'faESM2'
        else:
            raise ValueError(f"Unsupported plm style {plm_style}")

        self.PAD_TOKEN_IDX = 22 if self._plm_style == 'gLM2' else 23 if self._plm_style == 'faESM2' else None

        clean_pdb(self.pdb_raw, self.pdb_clean)
        self.data_pipeline()

    def data_pipeline(self):
        self.chain_meta_info = get_info_from_Bio(read_structure(self.pdb_clean))

        self.chain_condition = [k for k in self.chain_meta_info if k!= self.chain_design]
        self.chains = [self.chain_design] + self.chain_condition

        self.chain_AF_style = {}
        for chain_id in self.chains:
            self.chain_AF_style[chain_id] = chain_meta_info_to_AF_style(self.chain_meta_info[chain_id])

        self.chain_len_list_condition = [len(self.chain_meta_info[k]['seq']) for k in self.chain_condition]
        if self.sample_length == None:
            self.sample_length = len(self.chain_meta_info[self.chain_design]['seq'])
        self.chain_len_list = [self.sample_length] + self.chain_len_list_condition

        self.chain_APM_style = {}
        for idx, chain_id in enumerate(self.chains):
            self.chain_APM_style[chain_id] = {}
            self.chain_APM_style[chain_id]['aatype'] = self.chain_AF_style[chain_id]['aatype']
            self.chain_APM_style[chain_id]['atom_positions'] = self.chain_AF_style[chain_id]['atom_positions']
            self.chain_APM_style[chain_id]['atom_mask'] = self.chain_AF_style[chain_id]['atom_mask']
            self.chain_APM_style[chain_id]['residue_index'] = self.chain_AF_style[chain_id]['res_seq']
            self.chain_APM_style[chain_id]['chain_index'] = torch.tensor([idx] * len(self.chain_AF_style[chain_id]['aatype']))
            self.chain_APM_style[chain_id]['diffuse_mask'] = torch.zeros_like(self.chain_AF_style[chain_id]['aatype'])
            self.chain_APM_style[chain_id]['bb_positions'] = self.chain_AF_style[chain_id]['atom_positions'][:, 1, :]
            self.chain_APM_style[chain_id]['bb_mask'] = self.chain_AF_style[chain_id]['atom_mask'][:, 1]
            self.chain_APM_style[chain_id]['bfactor'] = self.chain_AF_style[chain_id]['b_factors']

        self.chain_APM_style_condition = {}
        self.chain_APM_style_condition['aatype'] = torch.cat([self.chain_APM_style[k]['aatype'] for k in self.chain_condition], dim=0)
        self.chain_APM_style_condition['atom_positions'] = torch.cat([self.chain_APM_style[k]['atom_positions'] for k in self.chain_condition], dim=0)
        self.chain_APM_style_condition['atom_mask'] = torch.cat([self.chain_APM_style[k]['atom_mask'] for k in self.chain_condition], dim=0)
        self.chain_APM_style_condition['residue_index'] = torch.cat([self.chain_APM_style[k]['residue_index'] for k in self.chain_condition], dim=0)
        self.chain_APM_style_condition['chain_index'] = torch.cat([self.chain_APM_style[k]['chain_index'] for k in self.chain_condition], dim=0)
        self.chain_APM_style_condition['diffuse_mask'] = torch.cat([self.chain_APM_style[k]['diffuse_mask'] for k in self.chain_condition], dim=0)
        self.chain_APM_style_condition['bb_positions'] = torch.cat([self.chain_APM_style[k]['bb_positions'] for k in self.chain_condition], dim=0)
        self.chain_APM_style_condition['bb_mask'] = torch.cat([self.chain_APM_style[k]['bb_mask'] for k in self.chain_condition], dim=0)
        self.chain_APM_style_condition['bfactor'] = torch.cat([self.chain_APM_style[k]['bfactor'] for k in self.chain_condition], dim=0)

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):

        # concat condition data
        condition_data = _process_csv_row(self.chain_APM_style_condition)
        condition_data['chain_idx'] += 1

        # centering
        center_condition = torch.mean(condition_data['trans_1'], dim=0) # [3]

        surf_res = du.find_binding_surface_residues(condition_data['trans_1'])
        center_surface = condition_data['trans_1'][surf_res]
        random_vector = self.random_coefficient * torch.nn.functional.normalize(torch.randn(3), dim=0)

        if self._dataset_cfg.direction_condition == None or self._dataset_cfg.direction_surface == None:
            weight_cond = np.random.beta(self._dataset_cfg.random_alpha, self._dataset_cfg.random_beta)
            weight_surf = 1 - weight_cond
        else:
            weight_cond = self._dataset_cfg.direction_condition
            weight_surf = self._dataset_cfg.direction_surface
        center = weight_cond * center_condition + weight_surf * center_surface + random_vector

        condition_data['trans_1'] -= center.unsqueeze(0)

        # prepare noisy data
        noisy_trans = _centered_gaussian(1, self.sample_length, 'cpu') * du.NM_TO_ANG_SCALE
        noisy_trans = noisy_trans[0]
        noisy_rotmats = _uniform_so3(1, self.sample_length, 'cpu')[0]
        noisy_aatypes = _masked_categorical(1, self.sample_length, 'cpu')[0]
        noisy_torsions = _uniform_torsion(1, self.sample_length, 'cpu')[0]
        noisy_chain_idx = torch.ones(self.sample_length, dtype=torch.int64)
        noisy_res_idx = torch.arange(self.sample_length, dtype=torch.int64) + 1
        noisy_diffuse_mask = torch.ones(self.sample_length, dtype=torch.int64)

        # concat noisy and condition data
        complex_data = {}
        complex_data['aatypes_1'] = torch.cat([noisy_aatypes, condition_data['aatypes_1']], dim=0)
        complex_data['rotmats_1'] = torch.cat([noisy_rotmats, condition_data['rotmats_1']], dim=0)
        complex_data['trans_1'] = torch.cat([noisy_trans, condition_data['trans_1']], dim=0)
        complex_data['torsions_1'] = torch.cat([noisy_torsions, condition_data['torsions_1']], dim=0)
        complex_data['res_mask'] = torch.ones_like(complex_data['aatypes_1'])
        complex_data['chain_idx'] = torch.cat([noisy_chain_idx, condition_data['chain_idx']], dim=0)
        complex_data['res_idx'] = torch.cat([noisy_res_idx, condition_data['res_idx']], dim=0)
        complex_data['diffuse_mask'] = torch.cat([noisy_diffuse_mask, condition_data['diffuse_mask']], dim=0)

        # prepare PLM template
        template_dict = du.create_sequence_templates(self.chain_len_list)
        if self._plm_style == 'gLM2':
            complex_data['template'] = template_dict['gLM_template']
            complex_data['template_mask'] = template_dict['gLM_template_mask']
            complex_data['chain_lengthes'] = torch.tensor([x+1 for x in self.chain_len_list])
        elif self._plm_style == 'faESM2':
            complex_data['template'] = template_dict['faESM2_template']
            complex_data['template_mask'] = template_dict['faESM2_template_mask']
            complex_data['chain_lengthes'] = torch.tensor([x+2 for x in self.chain_len_list])

        complex_data['center'] = center
        complex_data['idx'] = torch.tensor([idx])

        return complex_data


if __name__ == '__main__':
    import hydra

    def read_config(config_path="../configs", config_name="inference_conditional.yaml"):
        with hydra.initialize(config_path=config_path):
            cfg = hydra.compose(config_name=config_name)
        return cfg

    config = read_config()
    dataset = ConditionalDataset(dataset_cfg=config.conditional_dataset)

    print(dataset[0])


    
