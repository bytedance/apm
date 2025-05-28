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


import abc
import numpy as np
import pandas as pd
import logging
import tree
import torch
import random
import os

from glob import glob
from torch.utils.data import Dataset
from apm.data import utils as du
from openfold.data import data_transforms
from openfold.utils import rigid_utils
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from apm.data.full_atom_utils import atom37_to_torsion_angles
import psutil

SRC_PATH = {'PDB':'data_APM/pdb_monomer',
            'Multimer':'data_APM/pdb_multimer',
            'AFDB':'data_APM/afdb',
            'SWISSPROT':'swissprot_data'}

def _rog_filter(df, quantile):
    y_quant = pd.pivot_table(
        df,
        values='radius_gyration', 
        index='modeled_seq_len',
        aggfunc=lambda x: np.quantile(x, quantile)
    )
    x_quant = y_quant.index.to_numpy()
    y_quant = y_quant.radius_gyration.to_numpy()

    # Fit polynomial regressor
    poly = PolynomialFeatures(degree=4, include_bias=True)
    poly_features = poly.fit_transform(x_quant[:, None])
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y_quant)

    # Calculate cutoff for all sequence lengths
    max_len = df.modeled_seq_len.max()
    pred_poly_features = poly.fit_transform(np.arange(max_len)[:, None])
    # Add a little more.
    pred_y = poly_reg_model.predict(pred_poly_features) + 0.1

    row_rog_cutoffs = df.modeled_seq_len.map(lambda x: pred_y[x-1])
    return df[df.radius_gyration < row_rog_cutoffs]


def _length_filter(data_csv, min_res, max_res):
    return data_csv[
        (data_csv.modeled_seq_len >= min_res)
        & (data_csv.modeled_seq_len <= max_res)
    ]


def _plddt_percent_filter(data_csv, min_plddt_percent):
    return data_csv[data_csv.num_confident_plddt > min_plddt_percent]


def _max_coil_filter(data_csv, max_coil_percent):
    return data_csv[data_csv.coil_percent <= max_coil_percent]


def _mean_plddt_filter(data_csv, min_mean_plddt):
    return data_csv[data_csv.plddt_mean > min_mean_plddt]


def _process_csv_row_FAESM(processed_file_path, crop_size=512, anno=None, conditional_multimer_prop=0.0, conditional_multimer_ratio=0.0, train_packing_only=False):
    processed_feats = du.read_pkl(processed_file_path)
    processed_feats = du.parse_chain_feats(processed_feats)
    
    template_style = 'ESM'
    if 'gLM2' in anno:
        template_style = 'gLM'

    # Only take modeled residues.
    modeled_idx = processed_feats['modeled_idx']
    min_idx = np.min(modeled_idx)
    max_idx = np.max(modeled_idx)
    del processed_feats['modeled_idx']
    processed_feats = tree.map_structure(
        lambda x: x[min_idx:(max_idx+1)], processed_feats)

    # Run through OpenFold data transforms.
    chain_feats = {
        'aatype': torch.tensor(processed_feats['aatype']).long(),
        'all_atom_positions': torch.tensor(processed_feats['atom_positions']).double(),
        'all_atom_mask': torch.tensor(processed_feats['atom_mask']).double()
    }
    # crop if multimer and length > 384
    chain_idx = torch.from_numpy(processed_feats['chain_index'])
    n_res = chain_idx.shape[0]
    if not crop_size is None: # means the current sample is multimer
        chain_feats['asym_id'] = chain_idx
        crop_idx_bool, interface_mask, central_res = du.get_spatial_crop_idx_simplfy(chain_feats, crop_size=crop_size, interface_threshold=12.0)
    else:
        crop_idx_bool = torch.ones_like(chain_idx)
        interface_mask = torch.zeros_like(chain_idx).long()
        central_res = None
    
    crop_idx_bool = crop_idx_bool.bool()

    chain_feats = data_transforms.atom37_to_frames(chain_feats)
    chain_feats = data_transforms.make_atom14_masks(chain_feats)
    chain_feats = data_transforms.make_atom14_positions(chain_feats)

    rigids_1 = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]
    rotmats_1 = rigids_1.get_rots().get_rot_mats()
    trans_1 = rigids_1.get_trans()
    res_mask = torch.tensor(processed_feats['bb_mask']).int()

    torsion_angles_sin_cos, torsion_angles, torsion_angles_mask = \
        atom37_to_torsion_angles(chain_feats)
    torsions_1 = torsion_angles[:, -4:]

    # Re-number residue indices for each chain such that it starts from 1.
    # Randomize chain indices.
    _chain_idx = processed_feats['chain_index']
    res_idx = processed_feats['residue_index']
    new_res_idx = np.zeros_like(res_idx)
    new_chain_idx = np.zeros_like(res_idx)
    all_chain_idx = np.unique(chain_idx).tolist()

    N_chains = len(all_chain_idx)
    shuffled_chain_idx = [i+1 for i in range(N_chains)]
    random.shuffle(shuffled_chain_idx)
    if template_style == 'gLM':
        # one special token, <+>, for each chain
        speical_token_mask_0 = torch.zeros(n_res+N_chains).bool() # indicate <+>
    else:
        # two special tokens, <cls> and <eos>, for each chain
        speical_token_mask_0 = torch.zeros(n_res+N_chains*2).bool() # indicate <cls>
        speical_token_mask_1 = torch.zeros(n_res+N_chains*2).bool() # indicate <eos>
    
    run_cond = random.random() < conditional_multimer_prop
    # in conditional chain mode, at least one chain will be generated (the fixed generate chain), 
    # and the rest chains will have a probability to be set as the condition, which means will not be interpolated
    if central_res is None: # means monomer or multimer is not cropped, the the fixed genrate chain is selected randomly
        fixed_gen_chain_id = random.choice(all_chain_idx)
    else: # otherwise, the fixed generate chain will be chain where central res locates, which means the central res and the corresponding chain will always need to be generated
        fixed_gen_chain_id = _chain_idx[central_res]
    
    chain_lengthes = []
    speical_token_index = 0
    diffuse_mask = torch.ones_like(res_mask)
    diffuse_mask_0 = torch.zeros_like(res_mask)
    for i, chain_id in enumerate(all_chain_idx):
        chain_mask = (_chain_idx == chain_id).astype(int)
        chain_length = chain_mask.sum().item()
        
        if chain_id != fixed_gen_chain_id and run_cond:
            cond_chain = random.random() < conditional_multimer_ratio
            if cond_chain:
                diffuse_mask = diffuse_mask * (1-chain_mask) + diffuse_mask_0 * chain_mask
        
        speical_token_mask_0[speical_token_index] = True
        if template_style == 'gLM':
            speical_token_index += 1
            chain_lengthes.append(chain_length+1)
        else:
            EOS_index = speical_token_index + 1 + chain_length
            speical_token_mask_1[EOS_index] = True
            speical_token_index += 2
            chain_lengthes.append(chain_length+2)
        speical_token_index += chain_length

        chain_min_idx = np.min(res_idx + (1 - chain_mask) * 1e5).astype(int)
        new_res_idx = new_res_idx + (res_idx - chain_min_idx + 1) * chain_mask

        # Shuffle chain_index
        replacement_chain_id = shuffled_chain_idx[i]
        new_chain_idx = new_chain_idx + replacement_chain_id * chain_mask
    
    new_chain_idx = torch.from_numpy(new_chain_idx).int()
    new_res_idx = torch.from_numpy(new_res_idx).int()

    diffuse_mask_cropped = diffuse_mask[crop_idx_bool].int()
    res_mask_cropped = res_mask[crop_idx_bool]
    loss_mask = diffuse_mask_cropped * res_mask_cropped
    if loss_mask.sum() == 0:
        diffuse_mask = torch.ones_like(diffuse_mask)

    if torch.isnan(trans_1).any() or torch.isnan(rotmats_1).any():
        raise ValueError(f'Found NaNs in {processed_file_path}')

    feats = {
        'aatypes_1': chain_feats['aatype'],
        'rotmats_1': rotmats_1[crop_idx_bool],
        'trans_1': trans_1[crop_idx_bool],
        'torsions_1': torsions_1[crop_idx_bool], 
        'torsions_mask': torsion_angles_mask[:, -4:][crop_idx_bool], 
        'res_mask': res_mask_cropped,
        'res_mask_aa': res_mask,
        'diffuse_mask': diffuse_mask[crop_idx_bool].int(),
        'diffuse_mask_aa': diffuse_mask.int(),
        'chain_idx': new_chain_idx,
        'res_idx': new_res_idx[crop_idx_bool],
        'crop_idx': crop_idx_bool,
        'interface_mask': interface_mask[crop_idx_bool],
    }

    if train_packing_only:
        for k in ('rigidgroups_gt_frames', 
              'rigidgroups_alt_gt_frames', 
              'rigidgroups_gt_exists', 
              'atom14_gt_positions', 
              'atom14_alt_gt_positions', 
              'atom14_gt_exists', 
              'atom14_atom_is_ambiguous', 
              'atom14_alt_gt_exists'):

            feats[k] = chain_feats[k][crop_idx_bool]

        bb_torsion_1 = torsions_1[crop_idx_bool][:, :3]
        bb_torsion_mask = torsion_angles_mask[:, -4:][crop_idx_bool][:, :3]
        feats['bb_torsions_1'] = bb_torsion_1
        feats['bb_torsions_mask'] = bb_torsion_mask

    template = torch.zeros_like(speical_token_mask_0).int()
    if template_style == 'gLM':
        template[speical_token_mask_0] = 21 # there is 21 existed token (20 AAs + 1 UNK/MASK), the token index for <+> is 21
        template_mask = speical_token_mask_0
    else:
        template[speical_token_mask_0] = 21 # there is 21 existed token (20 AAs + 1 UNK/MASK), the token index for <cls> is 21
        template[speical_token_mask_1] = 22 # there is 21 existed token (20 AAs + 1 UNK/MASK), the token index for <eos> is 22
        template_mask = torch.logical_or(speical_token_mask_0, speical_token_mask_1)
    
    feats['template'] = template
    feats['template_mask'] = template_mask
    feats['chain_lengthes'] = torch.Tensor(chain_lengthes).int()
    
    return feats

def _add_plddt_mask(feats, plddt_threshold):
    feats['plddt_mask'] = torch.tensor(
        feats['res_plddt'] > plddt_threshold).int()


def _read_clusters(cluster_path, synthetic=False):
    pdb_to_cluster = {}
    with open(cluster_path, "r") as f:
        for i,line in enumerate(f):
            for chain in line.split(' '):
                if not synthetic:
                    pdb = chain.split('_')[0].strip()
                else:
                    pdb = chain.strip()
                pdb_to_cluster[pdb.upper()] = i
    return pdb_to_cluster


class BaseDataset(Dataset):
    def __init__(
            self,
            *,
            dataset_cfg,
            is_training,
            task,
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self.task = task
        self.raw_csv = pd.read_csv(self.dataset_cfg.csv_path)
        metadata_csv = self._filter_metadata(self.raw_csv)
        metadata_csv = metadata_csv.sort_values(
            'modeled_seq_len', ascending=False)
        if self._dataset_cfg.use_redesigned:
            self.redesigned_csv = pd.read_csv(self._dataset_cfg.redesigned_csv_path)
            metadata_csv = metadata_csv.merge(
                self.redesigned_csv, left_on='pdb_name', right_on='example')
            metadata_csv = metadata_csv[metadata_csv.best_rmsd < 2.0]
        if self._dataset_cfg.cluster_path is not None:
            pdb_to_cluster = _read_clusters(self._dataset_cfg.cluster_path, synthetic=True)
            def cluster_lookup(pdb):
                pdb = pdb.upper()
                if pdb not in pdb_to_cluster:
                    raise ValueError(f'Cluster not found for {pdb}')
                return pdb_to_cluster[pdb]
            metadata_csv['cluster'] = metadata_csv['pdb_name'].map(cluster_lookup)
        self._create_split(metadata_csv)
        self._cache = {}
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)

    @property
    def is_training(self):
        return self._is_training

    @property
    def dataset_cfg(self):
        return self._dataset_cfg
    
    def __len__(self):
        return len(self.csv)

    @abc.abstractmethod
    def _filter_metadata(self, raw_csv: pd.DataFrame) -> pd.DataFrame:
        pass

    def _create_split(self, data_csv):
        # Training or validation specific logic.
        if self.is_training:
            self.csv = data_csv
            self._log.info(
                f'Training: {len(self.csv)} examples')
        else:
            if self._dataset_cfg.max_eval_length is None:
                eval_lengths = data_csv.modeled_seq_len
            else:
                eval_lengths = data_csv.modeled_seq_len[
                    data_csv.modeled_seq_len <= self._dataset_cfg.max_eval_length
                ]
            all_lengths = np.sort(eval_lengths.unique())
            length_indices = (len(all_lengths) - 1) * np.linspace(
                0.0, 1.0, self.dataset_cfg.num_eval_lengths)
            length_indices = length_indices.astype(int)
            eval_lengths = all_lengths[length_indices]
            eval_csv = data_csv[data_csv.modeled_seq_len.isin(eval_lengths)]

            # Fix a random seed to get the same split each time.
            eval_csv = eval_csv.groupby('modeled_seq_len').sample(
                self.dataset_cfg.samples_per_eval_length,
                replace=True,
                random_state=123
            )
            eval_csv = eval_csv.sort_values('modeled_seq_len', ascending=False)
            self.csv = eval_csv
            self._log.info(
                f'Validation: {len(self.csv)} examples with lengths {eval_lengths}')
        self.csv['index'] = list(range(len(self.csv)))

    def process_csv_row(self, csv_row):
        path = csv_row['processed_path']

        if not self.actual_is_training:
            if csv_row.src == 'Multimer':
                path =  os.path.join('./multimer_unclustered/', os.path.basename(path))
            else:
                path = os.path.join('./pdb_test/', os.path.basename(path))
        else:
            src = csv_row.src
            assert src in SRC_PATH
            path = os.path.join(SRC_PATH[src], os.path.basename(path))

        seq_len = csv_row['modeled_seq_len']
        # Large protein files are slow to read. Cache them.
        use_cache = seq_len > self._dataset_cfg.cache_num_res
        if use_cache and path in self._cache:
            return self._cache[path]

        if 'multimer_crop_size' in self._dataset_cfg and csv_row.src == 'Multimer' and self.actual_is_training: # crop is only occurred in training
            crop_size = self._dataset_cfg.multimer_crop_size
        else:
            crop_size = None
        
        # processed_row = _process_csv_row(path, crop_size, crop_threshold, self.anno)
        processed_row = _process_csv_row_FAESM(path, crop_size, self.anno, self.conditional_multimer_prop, self.conditional_multimer_ratio, train_packing_only=self._dataset_cfg.train_packing_only)
        processed_row['pdb_name'] = csv_row['pdb_name']
        if self._dataset_cfg.use_redesigned:
            best_seq = csv_row['best_seq']
            if not isinstance(best_seq, float):
                best_aatype = torch.tensor(du.seq_to_aatype(best_seq)).long()
                assert processed_row['aatypes_1'].shape == best_aatype.shape
                processed_row['aatypes_1'] = best_aatype
        aatypes_1 = du.to_numpy(processed_row['aatypes_1'])
        if len(set(aatypes_1)) == 1:
            raise ValueError(f'Example {path} has only one amino acid.')
        curr_memory_used = psutil.virtual_memory().used/1024/1024/1024 # GB
        not_OOM = curr_memory_used < self.RAM_threshold # cache should exceed 90% of the total memory
        if use_cache and not_OOM:
            self._cache[path] = processed_row
        return processed_row
    
    
    def __getitem__(self, row_idx):
        # Process data example.
        csv_row = self.csv.iloc[row_idx]
        feats = self.process_csv_row(csv_row)

        if self._dataset_cfg.add_plddt_mask:
            _add_plddt_mask(feats, self._dataset_cfg.min_plddt_threshold)
        else:
            feats['plddt_mask'] = torch.ones_like(feats['res_mask'])
            
        return feats








def pdb_init_(
        self,
        *,
        dataset_cfg,
        is_training,
        task,
        anno=None,
    ):
    self._log = logging.getLogger(__name__)
    self._is_training = is_training
    self._dataset_cfg = dataset_cfg
    self.task = task
    self._cache = {}
    self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)
    self.RAM_threshold = self._dataset_cfg.max_memory * 0.9

    if 'mode' in self._dataset_cfg:
        self.actual_is_training = self._dataset_cfg.mode == 'train'
    else:
        self.actual_is_training = not os.path.basename(self._dataset_cfg.csv_path).startswith('test')

    if 'conditional_multimer_prop' in self._dataset_cfg:
        self.conditional_multimer_prop = self._dataset_cfg.conditional_multimer_prop
    else:
        self.conditional_multimer_prop = 0.0

    if 'conditional_multimer_ratio' in self._dataset_cfg:
        self.conditional_multimer_ratio = self._dataset_cfg.conditional_multimer_ratio
    else:
        self.conditional_multimer_ratio = 0.0

    if not self.actual_is_training:
        self.conditional_multimer_prop = 0.0
        self.conditional_multimer_ratio = 0.0
    
    ### The difference between conditional_multimer_prop and conditional_multimer_ratio is that:
    ###      conditional_multimer_prop means the proportion of conditional multimer examples
    ###      conditional_multimer_ratio means the ratio of chains within each sample to be set as conditional chain except the first chain

    if anno is None:
        self.anno = ''
    else:
        self.anno = anno

    self.multimer_metadata_csv_path = 'metadata_all/meta_raw_with_validtag.csv'
    self.multimer_max_length = self._dataset_cfg.multimer_length_threshold
    self.ignore_peptide = True

    ignore_multimer_ids = None
    with open('metadata_all/sabdab_summary_all_2024_10_26.csv') as f:
        sabadab_ids = [line[:4] for line in f]
        sabadab_ids = list(set(sabadab_ids[1:]))
        ignore_multimer_ids = sabadab_ids

    # ignore length less than 30 for peptide
    if self.ignore_peptide:
        excluded_pdb_names_l30 = torch.load("metadata_all/excluded_pdb_names_l30_.pt")
        excluded_pdb_names_l30 = [x[1:] for x in excluded_pdb_names_l30]
        ignore_multimer_ids += excluded_pdb_names_l30

    # Process clusters
    self.raw_csv = pd.read_csv(self.dataset_cfg.csv_path)
    
    use_multimer_data = False
    if 'use_multimer' in self._dataset_cfg:
        if self._dataset_cfg.use_multimer and self.actual_is_training:
            use_multimer_data = True

    use_multimer_data_only = False
    if 'use_multimer_only' in self._dataset_cfg:
        if self._dataset_cfg.use_multimer_only:
            use_multimer_data_only = True

    if use_multimer_data_only:
        self._dataset_cfg.use_redesigned = False
        self._dataset_cfg.use_synthetic = False
        self._dataset_cfg.use_AFDB = False
        self._dataset_cfg.use_SWISSPROT = False
        use_multimer_data = False
        if not self.actual_is_training:
            use_unclustered_multimer_only = True
        else:
            use_unclustered_multimer_only = False
        metadata_csv = self._gen_multimer_metadata(meta_csv_path=self.multimer_metadata_csv_path, 
                                                   max_length=self.multimer_max_length, 
                                                   ignore_pdb_ids=ignore_multimer_ids,
                                                   use_unclustered_multimer=self._dataset_cfg.use_unclustered_multimer,
                                                   use_unclustered_multimer_only=use_unclustered_multimer_only)
        
    else:
        metadata_csv = self._filter_metadata(self.raw_csv)
        metadata_csv['src'] = ['PDB', ] * len(metadata_csv)
        metadata_csv['num_chains'] = [1, ] * len(metadata_csv)
    
    metadata_csv = metadata_csv.sort_values(
        'modeled_seq_len', ascending=False)

    self._pdb_to_cluster = _read_clusters(self._dataset_cfg.cluster_path, synthetic=False)
    self._max_cluster = max(self._pdb_to_cluster.values())
    self._missing_pdbs = 0
    def cluster_lookup(pdb):
        pdb = pdb.upper()
        if pdb not in self._pdb_to_cluster:
            self._pdb_to_cluster[pdb] = self._max_cluster + 1
            self._max_cluster += 1
            self._missing_pdbs += 1
        return self._pdb_to_cluster[pdb]

    if self._dataset_cfg.use_redesigned:
        self.redesigned_csv = pd.read_csv(self._dataset_cfg.redesigned_csv_path)
        metadata_csv = metadata_csv.merge(
            self.redesigned_csv, left_on='pdb_name', right_on='example')
        metadata_csv = metadata_csv[metadata_csv.best_rmsd < 2.0]

    if use_multimer_data:
        multimer_meta_csv = self._gen_multimer_metadata(meta_csv_path=self.multimer_metadata_csv_path, 
                                                        max_length=self.multimer_max_length, 
                                                        ignore_pdb_ids=ignore_multimer_ids,
                                                        use_unclustered_multimer=self._dataset_cfg.use_unclustered_multimer)
        metadata_csv = pd.concat([metadata_csv, multimer_meta_csv])

    metadata_csv['cluster'] = metadata_csv['pdb_name'].map(cluster_lookup)
    
    if self._dataset_cfg.use_synthetic:
        self.synthetic_csv = pd.read_csv(self._dataset_cfg.synthetic_csv_path)
        self._synthetic_pdb_to_cluster = _read_clusters(self._dataset_cfg.synthetic_cluster_path, synthetic=True)

        # offset all the cluster numbers by the number of real data clusters
        num_real_clusters = metadata_csv['cluster'].max() + 1
        def synthetic_cluster_lookup(pdb):
            pdb = pdb.upper()
            if pdb not in self._synthetic_pdb_to_cluster:
                raise ValueError(f"Synthetic example {pdb} not in synthetic cluster file!")
            return self._synthetic_pdb_to_cluster[pdb] + num_real_clusters
        self.synthetic_csv['cluster'] = self.synthetic_csv['pdb_name'].map(synthetic_cluster_lookup)
        metadata_csv = pd.concat([metadata_csv, self.synthetic_csv])

    use_AFDB = False
    if 'use_AFDB' in self._dataset_cfg and self.actual_is_training:
        use_AFDB = self._dataset_cfg.use_AFDB

    if use_AFDB:
        AFDB_csv_path = 'metadata_all/metadata_90.csv'
        AFDB_cluster_path = 'metadata_all/AFDB.clusters'
        self.AFDB_csv = pd.read_csv(AFDB_csv_path)
        self._AFDB_pdb_to_cluster = _read_clusters(AFDB_cluster_path, synthetic=True)
        # offset all the cluster numbers by the number of real data clusters
        num_real_clusters = metadata_csv['cluster'].max() + 1
        def AFDB_cluster_lookup(pdb):
            pdb = pdb.upper()
            return self._AFDB_pdb_to_cluster[pdb] + num_real_clusters
        self.AFDB_csv['cluster'] = self.AFDB_csv['pdb_name'].map(AFDB_cluster_lookup)
        self.AFDB_csv = self._filter_AFDB_metadata(self.AFDB_csv)
        self.AFDB_csv['src'] = ['AFDB', ] * len(self.AFDB_csv)
        self.AFDB_csv['num_chains'] = [1, ] * len(self.AFDB_csv)
        metadata_csv = pd.concat([metadata_csv, self.AFDB_csv])
    
    use_SWISSPROT = False
    if 'use_SWISSPROT' in self._dataset_cfg and self.actual_is_training:
        use_SWISSPROT = self._dataset_cfg.use_SWISSPROT
    
    if use_SWISSPROT:
        SWISSPROT_meta_path = 'metadata_all/swissprot_metadata.csv'
        SWISSPROT_clu_path = 'metadata_all/swissprot_cluster50_cluster.tsv'
        SWISSPROT_csv = pd.read_csv(SWISSPROT_meta_path)
        SWISSPROT_csv = SWISSPROT_csv[SWISSPROT_csv.modeled_seq_len >= self.dataset_cfg.filter.min_num_res]
        SWISSPROT_csv = SWISSPROT_csv[SWISSPROT_csv.modeled_seq_len <= self.dataset_cfg.filter.max_num_res]
        SWISSPROT_csv = SWISSPROT_csv[SWISSPROT_csv.avg_plddt >= 85]
        SWISSPROT_csv = SWISSPROT_csv[SWISSPROT_csv.coil_percent <= self.dataset_cfg.filter.max_coil_percent]
        SWISSPROT_csv = _rog_filter(SWISSPROT_csv, self.dataset_cfg.filter.rog_quantile)

        SWISSPROT_clu_ = open(SWISSPROT_clu_path, 'r').readlines()
        SWISSPROT_clu_ = [x.strip().split('\t') for x in SWISSPROT_clu_]
        SWISSPROT_clu = {}
        for SWISSPROT_clu_id, SWISSPROT_sample_id in SWISSPROT_clu_:
            if SWISSPROT_clu_id not in SWISSPROT_clu:
                SWISSPROT_clu[SWISSPROT_clu_id] = [SWISSPROT_sample_id, ]
            else:
                SWISSPROT_clu[SWISSPROT_clu_id].append(SWISSPROT_sample_id)
        
        MAX_SWISSPROT_clu = len(SWISSPROT_clu) + 1
        curr_N_clusters = max(metadata_csv.cluster) + 1

        SWISSPROT_clu_mapping = {}
        for clu_index, clu_id in enumerate(SWISSPROT_clu):
            for sample_id in SWISSPROT_clu[clu_id]:
                SWISSPROT_clu_mapping[sample_id] = clu_index
        
        SWISSPROT_clu_info = []
        for sample_id in SWISSPROT_csv.pdb_name:
            if sample_id not in SWISSPROT_clu_mapping:
                sample_clu = MAX_SWISSPROT_clu + curr_N_clusters
                SWISSPROT_clu_info.append(sample_clu)
                MAX_SWISSPROT_clu += 1
            else:
                sample_clu = SWISSPROT_clu_mapping[sample_id] + curr_N_clusters
                SWISSPROT_clu_info.append(sample_clu)

        SWISSPROT_csv['cluster'] = SWISSPROT_clu_info
        existed_SWISSPROT_pkls = os.listdir(f'./swissprot_data')
        existed_SWISSPROT_pkls = [pkl.split('.')[0] for pkl in existed_SWISSPROT_pkls]
        SWISSPROT_csv = SWISSPROT_csv[SWISSPROT_csv['pdb_name'].isin(existed_SWISSPROT_pkls)]
        SWISSPROT_csv['src'] = ['SWISSPROT', ] * len(SWISSPROT_csv)
        SWISSPROT_csv['num_chains'] = [1, ] * len(SWISSPROT_csv)
        SWISSPROT_csv = SWISSPROT_csv[~SWISSPROT_csv['pdb_name'].isin(metadata_csv.pdb_name)]
        metadata_csv = pd.concat([metadata_csv, SWISSPROT_csv])

    self._create_split(metadata_csv)

    if dataset_cfg.test_set_pdb_ids_path is not None:

        test_set_df = pd.read_csv(dataset_cfg.test_set_pdb_ids_path)

        self.csv = self.csv[self.csv['pdb_name'].isin(test_set_df['pdb_name'].values)]

def pdb_filter_metadata(self, raw_csv):
    """Filter metadata."""
    filter_cfg = self.dataset_cfg.filter
    data_csv = raw_csv[
        raw_csv.oligomeric_detail.isin(filter_cfg.oligomeric)]
    data_csv = data_csv[
        data_csv.num_chains.isin(filter_cfg.num_chains)]
    data_csv = _length_filter(
        data_csv, filter_cfg.min_num_res, filter_cfg.max_num_res)
    data_csv = _max_coil_filter(data_csv, filter_cfg.max_coil_percent)
    data_csv = _rog_filter(data_csv, filter_cfg.rog_quantile)
    return data_csv

def pdb_multimer_filter_metadata(self, raw_csv):
    """Filter metadata for multimer data."""
    filtered_multimer_pkl = torch.load('metadata_all/filtered_mutichain_pkl_files.pt')
    filtered_multimer_pkl_id = [pkl.split('/')[-1] for pkl in filtered_multimer_pkl]
    is_filtered_multimer = [int(processed_file.split('/')[-1] in filtered_multimer_pkl_id) for processed_file in raw_csv['processed_path']]
    raw_csv['is_filtered_multimer'] = is_filtered_multimer
    data_csv = raw_csv[raw_csv.is_filtered_multimer==1]
    data_csv = _rog_filter(data_csv, 0.96)
    return data_csv

def pdb_AFDB_filter_metadata(self, data_csv):
    """Filter metadata for AFDB."""
    filter_cfg = self.dataset_cfg.filter
    data_csv = _length_filter(
        data_csv, filter_cfg.min_num_res, filter_cfg.max_num_res)
    data_csv = _max_coil_filter(data_csv, filter_cfg.max_coil_percent)
    data_csv = _rog_filter(data_csv, filter_cfg.rog_quantile)
    data_csv = _mean_plddt_filter(data_csv, filter_cfg.AFDB_plddt_threshold)
    return data_csv

def gen_multimer_metadata(meta_csv_path, max_length=768, ignore_pdb_ids=None, use_unclustered_multimer=False, use_unclustered_multimer_only=False):
    """Generate metadata for multimer data."""
    multimer_meta_csv = pd.read_csv(meta_csv_path)
    pdb_ids = [pdb_name[1:] if len(pdb_name)==5 else pdb_name for pdb_name in multimer_meta_csv['pdb_name']] # correct the pdb name, a # is behind the pdb id as some pdb ids like 12e8 will be incorrectly processed as a float number
    multimer_meta_csv['pdb_name'] = pdb_ids

    if not ignore_pdb_ids is None:
        multimer_meta_csv = multimer_meta_csv[~multimer_meta_csv['pdb_name'].isin(ignore_pdb_ids)]

    if use_unclustered_multimer_only:
        use_unclustered_multimer = True

    if not use_unclustered_multimer:
        multimer_meta_csv = multimer_meta_csv[multimer_meta_csv['cluster']>=0]
    
    if use_unclustered_multimer_only:
        multimer_meta_csv = multimer_meta_csv[multimer_meta_csv['cluster']<0]
    
    multimer_meta_csv = multimer_meta_csv[multimer_meta_csv['modeled_seq_len']<max_length]

    multimer_meta_csv = _rog_filter(multimer_meta_csv, 0.96)
    multimer_meta_csv.drop('cluster', axis=1)
    multimer_meta_csv['src'] = ['Multimer', ] * len(multimer_meta_csv)

    if not use_unclustered_multimer:
        for pdb_name in multimer_meta_csv['pdb_name']:
            assert os.path.exists(f'./data_APM/pdb_multimer/{pdb_name}.pkl')

    return multimer_meta_csv

class PdbDataset(BaseDataset):

    def __init__(self, *, dataset_cfg, is_training, task, anno=None):
        pdb_init_(self, dataset_cfg=dataset_cfg, is_training=is_training, task=task, anno=anno)

    def _filter_metadata(self, raw_csv):
        return pdb_filter_metadata(self, raw_csv)

    def _filter_multimer_metadata(self, raw_csv):
        return pdb_multimer_filter_metadata(self, raw_csv)
    
    def _gen_multimer_metadata(self, meta_csv_path, max_length=768, ignore_pdb_ids=None, use_unclustered_multimer=False, use_unclustered_multimer_only=False):
        return gen_multimer_metadata(meta_csv_path, max_length, ignore_pdb_ids=ignore_pdb_ids, use_unclustered_multimer=use_unclustered_multimer, use_unclustered_multimer_only=use_unclustered_multimer_only)

    def _filter_AFDB_metadata(self, raw_csv):
        return pdb_AFDB_filter_metadata(self, raw_csv)