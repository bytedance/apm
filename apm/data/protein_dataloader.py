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


"""Protein data loader."""
import math
import torch
import torch
import logging
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler, dist
from glob import glob
import os
import numpy as np
from apm.data import utils as du
from random import shuffle

def custom_group_by(data_csv, by_key, shuffle_key=True):
    group_dict = {}
    for key_value, sub_df in data_csv.groupby(by_key):
        group_dict[key_value] = sub_df
    
    if shuffle_key:
        key_list = list(group_dict.keys())
        shuffle(key_list)
        group_dict = [group_dict[key] for key in key_list]
    
    return group_dict

class ProteinData(LightningDataModule):

    def __init__(self, *, data_cfg, train_dataset, valid_dataset, dataset_cfg, predict_dataset=None, replica=True, equal_chain_nums=False):
        super().__init__()
        self.data_cfg = data_cfg
        self.loader_cfg = data_cfg.loader
        self.sampler_cfg = data_cfg.sampler
        self.dataset_cfg = dataset_cfg
        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset
        self._predict_dataset = predict_dataset
        self.replica = replica
        self.equal_chain_nums = equal_chain_nums

    def train_dataloader(self, rank=None, num_replicas=None):
        num_workers = self.loader_cfg.num_workers

        if self.replica:
            batch_sampler = LengthBatcher(
                sampler_cfg=self.sampler_cfg,
                metadata_csv=self._train_dataset.csv,
                rank=rank,
                num_replicas=num_replicas,
            )
        else:
            if self.equal_chain_nums:
                batch_sampler = LengthBatcher_nonRep_eqCN(
                    sampler_cfg=self.sampler_cfg,
                    metadata_csv=self._train_dataset.csv,
                    rank=rank,
                    num_replicas=num_replicas,
                )
            else:
                batch_sampler = LengthBatcher_nonRep(
                    sampler_cfg=self.sampler_cfg,
                    metadata_csv=self._train_dataset.csv,
                    rank=rank,
                    num_replicas=num_replicas,
                )

        return DataLoader(
            self._train_dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self._valid_dataset,
            sampler=DistributedSampler(self._valid_dataset, shuffle=False),
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        num_workers = self.loader_cfg.num_workers
        return DataLoader(
            self._predict_dataset,
            sampler=DistributedSampler(self._predict_dataset, shuffle=False),
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
            persistent_workers=True,
        )


class LengthBatcher:

    def __init__(
            self,
            *,
            sampler_cfg,
            metadata_csv,
            seed=123,
            shuffle=True,
            num_replicas=None,
            rank=None,
        ):
        super().__init__()
        self._log = logging.getLogger(__name__)
        if num_replicas is None:
            self.num_replicas = dist.get_world_size() # the number of all available GPUs
        else:
            self.num_replicas = num_replicas
        if rank is None:
            self.rank = dist.get_rank() # the current GPU id
        else:
            self.rank = rank

        self._sampler_cfg = sampler_cfg
        self._data_csv = metadata_csv

        # Each replica needs the same number of batches. We set the number
        # of batches to arbitrarily be the number of examples per replica.
        if 'cluster' in self._data_csv:
            num_batches = self._data_csv['cluster'].nunique() # for cluster is existed, num_batches is the number of all clusters
        else:
            num_batches = len(self._data_csv) # for cluster is not existed, num_batches is the number of all samples
        self.overall_num_batches = num_batches
        self._num_batches = math.ceil(self.overall_num_batches / self.num_replicas) # number of batches for each GPU
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.max_batch_size =  self._sampler_cfg.max_batch_size
        self._log.info(f'Created dataloader rank {self.rank+1} out of {self.num_replicas}')

    def _sample_indices(self):
        if 'cluster' in self._data_csv:
            cluster_sample = self._data_csv.groupby('cluster').sample(
                1, random_state=self.seed + self.epoch)
            return cluster_sample['index'].tolist() # random sample one sample from each cluster, and 
        else:
            return self._data_csv['index'].tolist()
        
    def _replica_epoch_batches(self, num_augments=None):
        # Make sure all replicas share the same seed on each epoch.
        rng = torch.Generator()
        if not num_augments is None:
            curr_seed = self.seed + self.epoch * self._num_batches + num_augments
        else:
            curr_seed = self.seed + self.epoch
        rng.manual_seed(curr_seed)
        indices = self._sample_indices() # random sampled indexes of cluster-1-samples (1 sample for each cluster)
        if self.shuffle:
            new_order = torch.randperm(len(indices), generator=rng).numpy().tolist()
            indices = [indices[i] for i in new_order]

        if len(self._data_csv) > self.num_replicas:
            replica_csv = self._data_csv.iloc[indices[self.rank::self.num_replicas]] # split the sampled cluster-1-samples into multiple replicas and get the corresponding replica
        else:
            replica_csv = self._data_csv
        
        # Each batch contains multiple proteins of the same length.
        sample_order = []

        for seq_len, len_df in replica_csv.groupby('modeled_seq_len'):
            cropped_seq_len = min(seq_len, 512) # the multimer exceeds 512 will be cropped to 512
            max_batch_size = min(
                self.max_batch_size,
                self._sampler_cfg.max_num_res_squared // cropped_seq_len**2 + 1,
            ) # determined the current batch size, since each batch only contains proteins of the same length
            num_batches = math.ceil(len(len_df) / max_batch_size) # the number of batches for the current length in this replica
            for i in range(num_batches):
                batch_df = len_df.iloc[i*max_batch_size:(i+1)*max_batch_size]
                batch_indices = batch_df['index'].tolist()
                batch_repeats = math.floor(max_batch_size / len(batch_indices)) # if the total number of samples with the same length is smaller than the current batch_size, then we need to repeat the samples to fill the batch_size
                sample_order.append(batch_indices * batch_repeats)
        
        # Remove any length bias.
        if self.shuffle:
            length_rng = torch.Generator()
            length_rng.manual_seed(self.seed + self.epoch + self.rank) # make the batch for each GPU within the same time is not with the same length
            new_order = torch.randperm(len(sample_order), generator=length_rng).numpy().tolist()
            return [sample_order[i] for i in new_order]
        return sample_order

    def _create_batches(self):
        # Make sure all replicas have the same number of batches Otherwise leads to bugs.
        # See bugs with shuffling https://github.com/Lightning-AI/lightning/issues/10947
        all_batches = []
        num_augments = 0
        while len(all_batches) < self._num_batches:
            all_batches.extend(self._replica_epoch_batches(num_augments))
            num_augments += 1
            if num_augments == 1000:
                raise ValueError('Exceeded number of augmentations.')
        if len(all_batches) >= self._num_batches:
            all_batches = all_batches[:self._num_batches]
        self.sample_order = all_batches

    def __iter__(self):
        self._create_batches()
        self.epoch += 1
        return iter(self.sample_order)

    def __len__(self):
        return self._num_batches

class LengthBatcher_nonRep:

    def __init__(
            self,
            *,
            sampler_cfg,
            metadata_csv,
            seed=123,
            shuffle=True,
            num_replicas=None,
            rank=None,
        ):
        super().__init__()
        self._log = logging.getLogger(__name__)
        if num_replicas is None:
            self.num_replicas = dist.get_world_size() # the number of all available GPUs
        else:
            self.num_replicas = num_replicas
        if rank is None:
            self.rank = dist.get_rank() # the current GPU id
        else:
            self.rank = rank

        self._sampler_cfg = sampler_cfg
        self._data_csv = metadata_csv

        # Each replica needs the same number of batches. We set the number
        # of batches to arbitrarily be the number of examples per replica.
        if 'cluster' in self._data_csv:
            num_batches = self._data_csv['cluster'].nunique() # for cluster is existed, num_batches is the number of all clusters
        else:
            num_batches = len(self._data_csv) # for cluster is not existed, num_batches is the number of all samples
        self.overall_num_batches = num_batches
        self._num_batches = math.ceil(self.overall_num_batches / self.num_replicas) # number of batches for each GPU
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.max_batch_size =  self._sampler_cfg.max_batch_size
        self._log.info(f'Created dataloader rank {self.rank+1} out of {self.num_replicas}')

    def _sample_indices(self):
        # get the all sample indexes of cluster-1-samples (1 sample for each cluster) for the current epoch for all GPUs
        if 'cluster' in self._data_csv:
            cluster_sample = self._data_csv.groupby('cluster').sample(
                1, random_state=self.seed + self.epoch)
            return cluster_sample['index'].tolist() # random sample one sample from each cluster, and 
        else:
            return self._data_csv['index'].tolist()
    
    def _epoch_batches(self, ):
        rng = torch.Generator()
        curr_seed = self.seed + self.epoch
        rng.manual_seed(curr_seed)
        indices = self._sample_indices() # random sampled indexes of cluster-1-samples (1 sample for each cluster)
        if self.shuffle:
            new_order = torch.randperm(len(indices), generator=rng).numpy().tolist()
            indices = [indices[i] for i in new_order]
        non_rep_csv = self._data_csv.iloc[indices]

        sample_order = []
        for seq_len, len_df in non_rep_csv.groupby('modeled_seq_len'):
            cropped_seq_len = min(seq_len, 384) # the multimer exceeds 384 will be cropped to 384
            max_batch_size = min(
                self.max_batch_size,
                self._sampler_cfg.max_num_res_squared // cropped_seq_len**2 + 1,
            ) # determined the current batch size, since each batch only contains proteins of the same length
            num_batches = math.ceil(len(len_df) / max_batch_size) # the number of batches for the current length in this replica
            for i in range(num_batches):
                batch_df = len_df.iloc[i*max_batch_size:(i+1)*max_batch_size]
                batch_indices = batch_df['index'].tolist()
                batch_repeats = math.floor(max_batch_size / len(batch_indices)) # if the total number of samples with the same length is smaller than the current batch_size, then we need to repeat the samples to fill the batch_size
                sample_order.append(batch_indices * batch_repeats)

        if self.shuffle:
            length_rng = torch.Generator()
            length_rng.manual_seed(curr_seed)
            new_order = torch.randperm(len(sample_order), generator=length_rng).numpy().tolist()
            sample_order = [sample_order[i] for i in new_order]
        
        if len(sample_order)%self.num_replicas!= 0:
            N_padding_batches = self.num_replicas - len(sample_order)%self.num_replicas
            padding_indices = np.random.choice(range(len(sample_order)), N_padding_batches)
            sample_order = sample_order + [sample_order[i] for i in padding_indices]

        # Split the sample_order into each GPU
        rank_sample_order = [sample_order[i] for i in range(self.rank, len(sample_order), self.num_replicas)]
        self._num_batches = len(rank_sample_order)
        return rank_sample_order

    def _create_batches(self):
        self.sample_order = self._epoch_batches()

    def __iter__(self):
        self._create_batches()
        self.epoch += 1
        return iter(self.sample_order)

    def __len__(self):
        return self._num_batches


class LengthBatcher_nonRep_eqCN:

    def __init__(
            self,
            *,
            sampler_cfg,
            metadata_csv,
            seed=123,
            shuffle=True,
            num_replicas=None,
            rank=None,
        ):
        super().__init__()
        self._log = logging.getLogger(__name__)
        if num_replicas is None:
            self.num_replicas = dist.get_world_size() # the number of all available GPUs
        else:
            self.num_replicas = num_replicas
        if rank is None:
            self.rank = dist.get_rank() # the current GPU id
        else:
            self.rank = rank

        self._sampler_cfg = sampler_cfg
        self._data_csv = metadata_csv

        # Each replica needs the same number of batches. We set the number
        # of batches to arbitrarily be the number of examples per replica.
        if 'cluster' in self._data_csv:
            num_batches = self._data_csv['cluster'].nunique() # for cluster is existed, num_batches is the number of all clusters
        else:
            num_batches = len(self._data_csv) # for cluster is not existed, num_batches is the number of all samples
        self.overall_num_batches = num_batches
        self._num_batches = math.ceil(self.overall_num_batches / self.num_replicas) # number of batches for each GPU
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.max_batch_size =  self._sampler_cfg.max_batch_size
        self._log.info(f'Created dataloader rank {self.rank+1} out of {self.num_replicas}')

    def _sample_indices(self):
        # get the all sample indexes of cluster-1-samples (1 sample for each cluster) for the current epoch for all GPUs
        if 'cluster' in self._data_csv:
            cluster_sample = self._data_csv.groupby('cluster').sample(
                1, random_state=self.seed + self.epoch)
            return cluster_sample['index'].tolist() # random sample one sample from each cluster, and 
        else:
            return self._data_csv['index'].tolist()
    
    def _epoch_batches(self, ):
        rng = torch.Generator()
        curr_seed = self.seed + self.epoch
        rng.manual_seed(curr_seed)
        indices = self._sample_indices() # random sampled indexes of cluster-1-samples (1 sample for each cluster)
        if self.shuffle:
            new_order = torch.randperm(len(indices), generator=rng).numpy().tolist()
            indices = [indices[i] for i in new_order]
        non_rep_csv = self._data_csv.iloc[indices]

        sample_order = []
        for seq_len, len_df in non_rep_csv.groupby('modeled_seq_len'):
            cropped_seq_len = min(seq_len, 384) # the multimer exceeds 384 will be cropped to 384
            max_batch_size = min(
                self.max_batch_size,
                self._sampler_cfg.max_num_res_squared // cropped_seq_len**2 + 1,
            ) # determined the current batch size, since each batch only contains proteins of the same length
            for chain_num, chain_df in len_df.groupby('num_chains'): # for gLM, each chain will be added a <+>, so the num_chains should be equal within the batch
                num_batches = math.ceil(len(chain_df) / max_batch_size) # the number of batches for the current length in this replica
                for i in range(num_batches):
                    batch_df = chain_df.iloc[i*max_batch_size:(i+1)*max_batch_size]
                    batch_indices = batch_df['index'].tolist()
                    batch_repeats = math.floor(max_batch_size / len(batch_indices)) # if the total number of samples with the same length is smaller than the current batch_size, then we need to repeat the samples to fill the batch_size
                    sample_order.append(batch_indices * batch_repeats)

        if self.shuffle:
            length_rng = torch.Generator()
            length_rng.manual_seed(curr_seed)
            new_order = torch.randperm(len(sample_order), generator=length_rng).numpy().tolist()
            sample_order = [sample_order[i] for i in new_order]
        
        if len(sample_order)%self.num_replicas!= 0:
            N_padding_batches = self.num_replicas - len(sample_order)%self.num_replicas
            padding_indices = np.random.choice(range(len(sample_order)), N_padding_batches)
            sample_order = sample_order + [sample_order[i] for i in padding_indices]

        # Split the sample_order into each GPU
        rank_sample_order = [sample_order[i] for i in range(self.rank, len(sample_order), self.num_replicas)]
        self._num_batches = len(rank_sample_order)
        return rank_sample_order

    def _create_batches(self):
        self.sample_order = self._epoch_batches()

    def __iter__(self):
        self._create_batches()
        self.epoch += 1
        return iter(self.sample_order)

    def __len__(self):
        return self._num_batches