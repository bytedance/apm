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


"""Utility functions for experiments."""
import os
import numpy as np
import random
import torch
import glob
import re
import GPUtil
import shutil
import subprocess
import pandas as pd
import torch.distributed as dist
from openfold.utils import rigid_utils
import logging
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from apm.analysis import utils as au
from openfold.utils import rigid_utils as ru
from biotite.sequence.io import fasta
from apm.utils.foldseek_align import call_foldseek_align
from tmtools import tm_align
from openfold.utils.superimposition import superimpose
from Bio import PDB
from Bio.SeqUtils import seq1

Rigid = rigid_utils.Rigid


class LengthDataset(torch.utils.data.Dataset):
    def __init__(self, samples_cfg):
        self._samples_cfg = samples_cfg
        all_sample_lengths = range(
            self._samples_cfg.min_length,
            self._samples_cfg.max_length+1,
            self._samples_cfg.length_step
        )
        if samples_cfg.length_subset is not None:
            all_sample_lengths = [
                int(x) for x in samples_cfg.length_subset
            ]
        all_sample_ids = []
        num_batch = self._samples_cfg.num_batch
        assert self._samples_cfg.samples_per_length % num_batch == 0
        self.n_samples = self._samples_cfg.samples_per_length // num_batch

        for length in all_sample_lengths:
            for sample_id in range(self.n_samples):
                sample_ids = torch.tensor([num_batch * sample_id + i for i in range(num_batch)])
                all_sample_ids.append((length, sample_ids))
        self._all_sample_ids = all_sample_ids

    def __len__(self):
        return len(self._all_sample_ids)

    def __getitem__(self, idx):
        num_res, sample_id = self._all_sample_ids[idx]
        batch = {
            'num_res': num_res,
            'sample_id': sample_id,
        }
        return batch

class LengthDataset_multimer(torch.utils.data.Dataset):
    def __init__(self, samples_cfg):
        self._samples_cfg = samples_cfg
        all_sample_lengths = [
            [int(0)]+[int(b) for b in x] for x in samples_cfg.length_subset
        ]
        all_sample_ids = []
        num_batch = self._samples_cfg.num_batch
        assert self._samples_cfg.samples_per_length % num_batch == 0
        self.n_samples = self._samples_cfg.samples_per_length // num_batch

        for length in all_sample_lengths:
            for sample_id in range(self.n_samples):
                sample_ids = torch.tensor([num_batch * sample_id + i for i in range(num_batch)])
                all_sample_ids.append((length, sample_ids))
        self._all_sample_ids = all_sample_ids

    def __len__(self):
        return len(self._all_sample_ids)

    def __getitem__(self, idx):
        num_res, sample_id = self._all_sample_ids[idx]
        batch = {
            'num_res': torch.Tensor(num_res),
            'sample_id': sample_id,
        }
        return batch

def dataset_creation(dataset_class, cfg, task, anno=None):
    train_dataset = dataset_class(
        dataset_cfg=cfg,
        task=task,
        is_training=True,
        anno=anno,
    ) 
    eval_dataset = dataset_class(
        dataset_cfg=cfg,
        task=task,
        is_training=False,
        anno=anno,
    ) 
    return train_dataset, eval_dataset


def get_available_device(num_device):
    return GPUtil.getAvailable(order='memory', limit = 8)[:num_device]

def run_easy_cluster(designable_dir, output_dir):
    # designable_dir should be a directory with individual PDB files in it that we want to cluster
    # output_dir is where we are going to save the easy cluster output files

    # Returns the number of clusters

    easy_cluster_args = [
        'foldseek',
        'easy-cluster',
        designable_dir,
        os.path.join(output_dir, 'res'),
        output_dir,
        '--alignment-type',
        '1',
        '--cov-mode',
        '0',
        '--min-seq-id',
        '0',
        '--tmscore-threshold',
        '0.5',
    ]
    process = subprocess.Popen(
        easy_cluster_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, _ = process.communicate()
    del stdout # We don't actually need the stdout, we will read the number of clusters from the output files
    rep_seq_fasta = fasta.FastaFile.read(os.path.join(output_dir, 'res_rep_seq.fasta'))
    return len(rep_seq_fasta)
 


def get_all_top_samples(output_dir, csv_fname='*/*/top_sample.csv'):
    all_csv_paths = glob.glob(os.path.join(output_dir, csv_fname), recursive=True)
    top_sample_csv = pd.concat([pd.read_csv(x) for x in all_csv_paths])
    top_sample_csv.to_csv(
        os.path.join(output_dir, 'all_top_samples.csv'), index=False)
    return top_sample_csv


def calculate_diversity(output_dir, metrics_df, top_sample_csv, designable_csv_path):
    designable_samples = top_sample_csv[top_sample_csv.designable]
    designable_dir = os.path.join(output_dir, 'designable')
    os.makedirs(designable_dir, exist_ok=True)
    designable_txt = os.path.join(designable_dir, 'designable.txt')
    if os.path.exists(designable_txt):
        os.remove(designable_txt)
    with open(designable_txt, 'w') as f:
        for _, row in designable_samples.iterrows():
            sample_path = row.sample_path
            sample_name = f'sample_id_{row.sample_id}_length_{row.length}.pdb'
            write_path = os.path.join(designable_dir, sample_name)
            shutil.copy(sample_path, write_path)
            f.write(write_path+'\n')
    if metrics_df['Total codesignable'].iloc[0] <= 1:
        metrics_df['Clusters'] = metrics_df['Total codesignable'].iloc[0]
    else:
        add_diversity_metrics(designable_dir, metrics_df, designable_csv_path)


def add_diversity_metrics(designable_dir, designable_csv, designable_csv_path):
    designable_txt = os.path.join(designable_dir, 'designable.txt')
    clusters = run_easy_cluster(designable_dir, designable_dir)
    designable_csv['Clusters'] = clusters
    designable_csv.to_csv(designable_csv_path, index=False)


def calculate_pmpnn_consistency(output_dir, designable_csv, designable_csv_path):
    # output dir points to directory containing length_60, length_61, ... etc folders
    sample_dirs = glob.glob(os.path.join(output_dir, 'length_*/sample_*'))
    average_accs = []
    max_accs = []
    for sample_dir in sample_dirs:
        sample_pdb = glob.glob(os.path.join(sample_dir, '*.pdb'))[0]
        sample_pdb = os.path.basename(sample_pdb)
        sample_pdb_id = sample_pdb[:-4]
        pmpnn_fasta_path = os.path.join(sample_dir, 'self_consistency', 'seqs', f'{sample_pdb_id}_modified.fasta')
        codesign_fasta_path = os.path.join(sample_dir, 'self_consistency', 'codesign_seqs', 'codesign.fa')
        pmpnn_fasta = fasta.FastaFile.read(pmpnn_fasta_path)
        codesign_fasta = fasta.FastaFile.read(codesign_fasta_path)
        codesign_seq = codesign_fasta['codesign_seq_1']
        codesign_seq = codesign_seq.replace('/', '') # in multimer mode, "/" is used to separate chains
        seq_length = len(codesign_seq)
        accs = []
        for seq in pmpnn_fasta:
            pmpnn_seq = pmpnn_fasta[seq]
            pmpnn_seq = pmpnn_seq.replace('/', '')
            pmpnn_seq_length = len(pmpnn_seq)
            assert pmpnn_seq_length == seq_length
            acc = np.mean([int(pmpnn_res == codesign_res) for pmpnn_res, codesign_res in zip(pmpnn_seq, codesign_seq)])
            # num_matches = sum([1 if pmpnn_fasta[seq][i] == codesign_seq[i] else 0 for i in range(seq_length)])
            # total_length = len(pmpnn_fasta[seq])
            # accs.append(num_matches / total_length)
            accs.append(acc)
        average_accs.append(np.mean(accs))
        max_accs.append(np.max(accs))
    designable_csv['Average PMPNN Consistency'] = np.mean(average_accs)
    designable_csv['Average Max PMPNN Consistency'] = np.mean(max_accs)
    designable_csv.to_csv(designable_csv_path, index=False)

def calculate_pmpnn_designability(output_dir, designable_csv, designable_csv_path):
    sample_dirs = glob.glob(os.path.join(output_dir, 'length_*/sample_*'))
    try:
        single_pmpnn_results = []
        top_pmpnn_results = []
        for sample_dir in sample_dirs:
            all_pmpnn_folds_df = pd.read_csv(os.path.join(sample_dir, 'pmpnn_results.csv'))
            single_pmpnn_fold_df = all_pmpnn_folds_df.iloc[[0]]
            single_pmpnn_results.append(single_pmpnn_fold_df)
            min_index = all_pmpnn_folds_df['bb_rmsd'].idxmin()
            top_pmpnn_df = all_pmpnn_folds_df.loc[[min_index]]
            top_pmpnn_results.append(top_pmpnn_df)
        single_pmpnn_results_df = pd.concat(single_pmpnn_results, ignore_index=True)
        top_pmpnn_results_df = pd.concat(top_pmpnn_results, ignore_index=True)
        designable_csv['Single seq PMPNN Designability'] = np.mean(single_pmpnn_results_df['bb_rmsd'].to_numpy() < 2.0)
        designable_csv['Top seq PMPNN Designability'] = np.mean(top_pmpnn_results_df['bb_rmsd'].to_numpy() < 2.0)
        designable_csv.to_csv(designable_csv_path, index=False)
    except:
        # TODO i think it breaks when one process gets here first
        print("calculate pmpnn designability didnt work")

def calculate_foldseek_novelty(output_dir, designable_csv, designable_csv_path, foldseek_db='foldseek_pdb_db'):
    sample_pdbs = glob.glob(os.path.join(output_dir, 'designable/*.pdb')) # only calculate the novelty for the designable samples
    N_pdbs = len(sample_pdbs)
    sample_alns = call_foldseek_align(sample_pdbs, foldseek_db)
    average_tmscore = [max(sample_alns[sample]['alntmscore']) for sample in sample_alns]
    N_missing_pdbs = N_pdbs - len(average_tmscore)
    if N_missing_pdbs > 0:
        average_tmscore += [0, ] * N_missing_pdbs
        print(f'****** {N_missing_pdbs} samples are missing in FoldSeek output, and corresponding TM-Score will be filled with 0.0 ******')
    designable_csv['Novelty'] = np.mean(average_tmscore)
    designable_csv.to_csv(designable_csv_path, index=False)

def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def get_ddp_info():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    node_id = rank // world_size
    return {"node_id": node_id, "local_rank": local_rank, "rank": rank, "world_size": world_size}


def flatten_dict(raw_dict):
    """Flattens a nested dict."""
    flattened = []
    for k, v in raw_dict.items():
        if isinstance(v, dict):
            flattened.extend([
                (f'{k}:{i}', j) for i, j in flatten_dict(v)
            ])
        else:
            flattened.append((k, v))
    return flattened


def save_traj(
        sample: np.ndarray,
        sample_full: np.ndarray,
        bb_prot_traj: np.ndarray,
        full_atom_prot_traj: np.ndarray,
        x0_traj: np.ndarray,
        x0_full_traj: np.ndarray,
        diffuse_mask: np.ndarray,
        output_dir: str,
        aa_traj = None,
        clean_aa_traj = None,
        chain_index = None, 
        write_trajectories = True,
        postfix = '',
    ):
    """Writes final sample and reverse diffusion trajectory.

    Args:
        bb_prot_traj: [noisy_T, N, 37, 3] atom37 sampled diffusion states.
            T is number of time steps. First time step is t=eps,
            i.e. bb_prot_traj[0] is the final sample after reverse diffusion.
            N is number of residues.
        x0_traj: [clean_T, N, 37, 3] atom37 predictions of clean data at each time step.
        res_mask: [N] residue mask.
        diffuse_mask: [N] which residues are diffused.
        output_dir: where to save samples.
        aa_traj: [noisy_T, N] amino acids (0 - 20 inclusive).
        clean_aa_traj: [clean_T, N] amino acids (0 - 20 inclusive).
        write_trajectories: bool Whether to also write the trajectories as well
                                 as the final sample

    Returns:
        Dictionary with paths to saved samples.
            'sample_path': PDB file of final state of reverse trajectory.
            'traj_path': PDB file os all intermediate diffused states.
            'x0_traj_path': PDB file of C-alpha x_0 predictions at each state.
        b_factors are set to 100 for diffused residues
        residues if there are any.
    """

    # Write sample.
    diffuse_mask = diffuse_mask.astype(bool)
    sample_path = os.path.join(output_dir, f'sample{postfix}.pdb')
    sample_full_path = os.path.join(output_dir, f'sample{postfix}_full.pdb')
    prot_traj_path = os.path.join(output_dir, 'bb_traj.pdb')
    prot_full_traj_path = os.path.join(output_dir, 'full_atom_traj.pdb')
    x0_traj_path = os.path.join(output_dir, 'x0_traj.pdb')
    x0_full_traj_path = os.path.join(output_dir, 'x0_full_traj.pdb')

    # Use b-factors to specify which residues are diffused.
    b_factors = np.tile((diffuse_mask * 100)[:, None], (1, 37))

    noisy_traj_length, num_res, _, _ = bb_prot_traj.shape
    clean_traj_length = x0_traj.shape[0]
    assert sample.shape == (num_res, 37, 3)
    assert bb_prot_traj.shape == (noisy_traj_length, num_res, 37, 3)
    assert x0_traj.shape == (clean_traj_length, num_res, 37, 3)

    if aa_traj is not None:
        assert aa_traj.shape == (noisy_traj_length, num_res)
        assert clean_aa_traj is not None
        assert clean_aa_traj.shape == (clean_traj_length, num_res)

    sample_path = au.write_prot_to_pdb(
        sample,
        sample_path,
        b_factors=b_factors,
        no_indexing=True,
        aatype=aa_traj[-1] if aa_traj is not None else None,
        chain_index=chain_index,
    )

    if sample_full.sum() != 0:
        sample_full_path = au.write_prot_to_pdb(
            sample_full,
            sample_full_path,
            b_factors=b_factors,
            no_indexing=True,
            aatype=aa_traj[-1] if aa_traj is not None else None,
            chain_index=chain_index,
        )

    if write_trajectories:
        prot_traj_path = au.write_prot_to_pdb(
            bb_prot_traj,
            prot_traj_path,
            b_factors=b_factors,
            no_indexing=True,
            aatype=aa_traj,
            chain_index=chain_index,
        )
        if full_atom_prot_traj.sum() != 0:
            prot_full_traj_path = au.write_prot_to_pdb(
                full_atom_prot_traj,
                prot_full_traj_path,
                b_factors=b_factors,
                no_indexing=True,
                aatype=aa_traj,
                chain_index=chain_index,
            )
        x0_traj_path = au.write_prot_to_pdb(
            x0_traj,
            x0_traj_path,
            b_factors=b_factors,
            no_indexing=True,
            aatype=clean_aa_traj,
            chain_index=chain_index,
        )
        if x0_full_traj.sum!= 0:
            x0_full_traj_path = au.write_prot_to_pdb(
                x0_full_traj,
                x0_full_traj_path,
                b_factors=b_factors,
                no_indexing=True,
                aatype=clean_aa_traj,
                chain_index=chain_index,
            )
    return {
        'sample_path': sample_path,
        'sample_full_path': sample_full_path,
        'traj_path': prot_traj_path,
        'prot_full_traj_path': prot_full_traj_path,
        'x0_traj_path': x0_traj_path,
        'x0_full_traj_path': x0_full_traj_path,
    }


def get_dataset_cfg(cfg):
    if cfg.data.dataset == 'pdb':
        return cfg.pdb_dataset
    else:
        raise ValueError(f'Unrecognized dataset {cfg.data.dataset}')


def get_sequences_from_pdb(pdb_file):
    """Extract amino acid sequences for each chain in a PDB file"""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    sequences = {}
    for model in structure:
        for chain in model:
            seq = ""
            for residue in chain:
                # if PDB.is_aa(residue, standard=False):
                    # try:
                aa = seq1(residue.resname)
                seq += aa
                    # except KeyError:
                    #     seq += 'X'
            if seq:
                sequences[chain.id] = seq
    return sequences

def calculate_bb_rmsd_and_tmsc(pdb1_path, pdb2_path):
    """
    Calculate bb_rmsd and bb_tmsc (based on CA atoms) between two PDB files

    Args:
        pdb1_path (str): Path to PDB file 1
        pdb2_path (str): Path to PDB file 2

    Returns:
        dict: Contains bb_rmsd, bb_tmsc and RMSD for each chain
    """
    # Extract CA atom coordinates (one atom per residue)
    def extract_ca_positions_by_chain(structure):
        chain_positions = {}
        for model in structure:
            for chain in model:
                ca_positions = []
                for residue in chain:
                    if 'CA' in residue:
                        ca_positions.append(residue['CA'].get_coord())
                chain_positions[chain.id] = np.array(ca_positions).reshape(-1, 3)
        return chain_positions

    # Read structures and extract CA coordinates
    parser = PDB.PDBParser(QUIET=True)
    structure1 = parser.get_structure('struct1', pdb1_path)
    structure2 = parser.get_structure('struct2', pdb2_path)
    ca_pos1_by_chain = extract_ca_positions_by_chain(structure1)
    ca_pos2_by_chain = extract_ca_positions_by_chain(structure2)

    # Extract sequences (each residue corresponds to one CA atom)
    seq1_by_chain = get_sequences_from_pdb(pdb1_path)
    seq2_by_chain = get_sequences_from_pdb(pdb2_path)

    # Check if residue counts match
    ca_pos1_all = np.concatenate(list(ca_pos1_by_chain.values()), axis=0)
    ca_pos2_all = np.concatenate(list(ca_pos2_by_chain.values()), axis=0)
    if len(ca_pos1_all) != len(ca_pos2_all):
        raise ValueError(
            f"Residue counts don't match: structure 1 has {len(ca_pos1_all)} CA atoms, structure 2 has {len(ca_pos2_all)} CA atoms"
        )
    # print(ca_pos1_all.shape, ca_pos2_all.shape)

    # Calculate overall bb_rmsd
    res_mask_all = torch.ones(len(ca_pos1_all))
    bb_rmsd_all = superimpose(
        torch.tensor(ca_pos1_all)[None],
        torch.tensor(ca_pos2_all)[None],
        res_mask_all
    )[1].item()

    # Extract sequences and check lengths
    seq1_all = ''.join(seq1_by_chain.values())
    seq2_all = ''.join(seq2_by_chain.values())
    if len(seq1_all) != len(ca_pos1_all) or len(seq2_all) != len(ca_pos2_all):
        raise ValueError(
            f"Sequence length doesn't match CA atom count, please check PDB file integrity, length1: {len(seq1_all)}, length2: {len(seq2_all)}"
        )

    # Calculate bb_tmsc (based on CA atoms)
    tm_result = tm_align(
        np.float64(ca_pos1_all),
        np.float64(ca_pos2_all),
        seq1_all,
        seq2_all
    )
    bb_tmsc_all = tm_result.tm_norm_chain1

    # Calculate RMSD for each chain
    chains_rmsd = {}
    for chain_id in ca_pos1_by_chain:
        if chain_id in ca_pos2_by_chain:
            pos1 = ca_pos1_by_chain[chain_id]
            pos2 = ca_pos2_by_chain[chain_id]
            res_mask = torch.ones(len(pos1))
            rmsd = superimpose(
                torch.tensor(pos1)[None],
                torch.tensor(pos2)[None],
                res_mask
            )[1].item()
            chains_rmsd[chain_id] = rmsd

    return {
        "bb_rmsd": bb_rmsd_all,
        "bb_tmsc": bb_tmsc_all,
        "length1": len(ca_pos1_all),
        "length2": len(ca_pos2_all),
        "chains_rmsd": chains_rmsd
    }


def calculate_multimer_folding_metrics(output_dir):
    rmsd_list = []
    tmsc_list = []
    chains_rmsd_list = []

    all_pdb_dir = glob.glob(os.path.join(output_dir, "length_*/*"))

    for multimer_dir in all_pdb_dir:
        pdb_name = os.path.basename(multimer_dir)
        gt_pdb = glob.glob(multimer_dir + "/*gt_1.pdb")[0]
        sample_pdb = glob.glob(multimer_dir + f"/*{pdb_name}.pdb")[0]
        
        record_dict = calculate_bb_rmsd_and_tmsc(gt_pdb, sample_pdb)
        bb_rmsd = record_dict['bb_rmsd']
        bb_tmsc = record_dict['bb_tmsc']
        chains_rmsd = record_dict['chains_rmsd']

        rmsd_list.append(bb_rmsd)
        tmsc_list.append(bb_tmsc)
        for chain_id, chains_rmsd in chains_rmsd.items():
            chains_rmsd_list.append(chains_rmsd)

    return {
        "bb_rmsd": rmsd_list,
        "bb_tmsc": tmsc_list,
        "chains_rmsd": chains_rmsd_list
    }


def indexed_path(base_path, pdb_name):
    idx = 0
    while True:
        file_name = f"{pdb_name}_{idx}.pdb"
        full_path = os.path.join(base_path, file_name)
        if not os.path.exists(full_path):
            return full_path
        idx += 1
