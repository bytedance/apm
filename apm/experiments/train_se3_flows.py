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


import os
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.plugins import CheckpointIO
from apm.data.datasets import PdbDataset
from apm.data.protein_dataloader import ProteinData
from apm.models.flow_module import FlowModule
from apm.experiments import utils as eu
import wandb

log = eu.get_pylogger(__name__)
torch.set_float32_matmul_precision('high')

def eight(n):
    return int(8*round(n/8))

def exp2(n):
    return int(np.exp2(round(np.log2(n))))

def GPU_MEMORY_MAPPING(curr_gpu_memory_size):
    if int(curr_gpu_memory_size) == 80:
        max_batch_size, max_num_res_squared = 100, 800000
    elif int(curr_gpu_memory_size) == 40:
        max_batch_size, max_num_res_squared = 42, 200000
    else:
        scale = float(curr_gpu_memory_size)/80
        max_batch_size = int(scale * 100)
        max_num_res_squared = int(scale**2 * 800000)
    return max_batch_size, max_num_res_squared

PLM_models = {'ESM-8M': 'esm2_t6_8M_UR50D.pt', 
              'ESM-35M': 'esm2_t12_35M_UR50D.pt', 
              'ESM-150M': 'esm2_t30_150M_UR50D.pt', 
              'ESM-650M': 'esm2_t33_650M_UR50D.pt', 
              'ESM-3B': 'esm2_t36_3B_UR50D.pt', 
              'DPLM-150M': 'dplm_150m', 
              'DPLM-650M': 'dplm_650m',
              'gLM2-150M': 'gLM2_150M',
              'gLM2-650M': 'gLM2_650M',
              'faESM2-650M': 'faESM2_650M',
              'faESMC-600M': 'faESMC_600M',}

class CustomCheckpointIO(CheckpointIO):
    @rank_zero_only
    def save_checkpoint(self, checkpoint, path, storage_options=None):

        curr_step = checkpoint['global_step']
        ckpt_to_save = {k: v for k, v in checkpoint.items() if k not in ['state_dict', 'optimizer_states']}
        selected_params = {k: v for k, v in checkpoint['state_dict'].items() if not k.startswith('folding')}
        ckpt_to_save['state_dict'] = selected_params

        if curr_step % 50000 == 0:
            ckpt_to_save['optimizer_states'] = checkpoint['optimizer_states']
        torch.save(ckpt_to_save, path)

    def load_checkpoint(self, path, map_location=None):
        return torch.load(path, map_location=map_location)

    def remove_checkpoint(self, path):
        os.remove(path)

class Experiment:

    def __init__(self, *, cfg: DictConfig):
        self._cfg = cfg
        self._data_cfg = cfg.data
        self._exp_cfg = cfg.experiment
        self._task = self._data_cfg.task
        self._dataset_cfg = self._setup_dataset()

        if hasattr(self._exp_cfg, 'device_memory'):
            self._data_cfg.sampler.max_batch_size, self._data_cfg.sampler.max_num_res_squared = GPU_MEMORY_MAPPING(self._exp_cfg.device_memory)
            self._data_cfg.sampler.max_batch_size = int(self._data_cfg.sampler.max_batch_size)
            self._data_cfg.sampler.max_num_res_squared = int(self._data_cfg.sampler.max_num_res_squared)

        equal_chain_nums = True

        self._datamodule: LightningDataModule = ProteinData(
            data_cfg=self._data_cfg,
            dataset_cfg=self._dataset_cfg,
            train_dataset=self._train_dataset,
            valid_dataset=self._valid_dataset,
            replica=self._exp_cfg.replica_batch,
            equal_chain_nums=equal_chain_nums,
        )
        total_devices = self._exp_cfg.num_devices
        if self._cfg.folding.own_device:
            total_devices += 1
        device_ids = eu.get_available_device(total_devices)
        
        folding_device_id = None
        self._train_device_ids = -1
        log.info(f"Training with devices: {self._train_device_ids}")

        reload_state_dict = None
        if self._exp_cfg.raw_state_dict_reload is not None:
            reload_state_ckpts = self._exp_cfg.raw_state_dict_reload.split('|')
            reload_state_ckpts = [ckpt.strip(' ') for ckpt in reload_state_ckpts]
            for reload_ckpt_path in reload_state_ckpts:
                reload_ckpt = torch.load(reload_ckpt_path, map_location='cpu', weights_only=False)
                reload_model_training_schedule = reload_ckpt['hyper_parameters']['cfg']['experiment']['training']['model_training_steps']
                reload_mode_types = [block.split('_')[0].split(',') for block in reload_model_training_schedule.split('-')]
                reload_mode_types = sum(reload_mode_types, [])
                if 'backbone' in reload_mode_types:
                    use_PLM = reload_ckpt['hyper_parameters']['cfg']['folding'].get('PLM', None)
                for model_type in reload_mode_types:
                    log.info(f'Reload {model_type} model from {reload_ckpt_path}')
                if reload_state_dict is None:
                    reload_state_dict = reload_ckpt['state_dict']
                else:
                    reload_state_dict.update(reload_ckpt['state_dict'])

            if use_PLM in list(PLM_models.values()) and use_PLM != self._cfg.folding.PLM:
                log.info(f'Following the reloaded ckpt, will use PLM:{use_PLM} to encode the sequence')
                self._cfg.folding.PLM = use_PLM
                
            if self._exp_cfg.train_packing_only == False:
                use_PLM_in_sidechain_model = True
                if use_PLM_in_sidechain_model:
                    log.info(f'will use PLM:{use_PLM} in sidechain model')
                self._cfg.packing_model.use_plm = use_PLM_in_sidechain_model
            else:
                self._cfg.packing_model.use_plm = False

        self._module: LightningModule = FlowModule(
            self._cfg,
            self._dataset_cfg,
            folding_cfg=self._cfg.folding,
            folding_device_id=folding_device_id,
        )
        
        if reload_state_dict is not None:
            self._module.load_state_dict(reload_state_dict, strict=False) # strict=False to support the current model load ckpt with another model has different design level
            reload_ckpts_str = ' & '.join(reload_state_ckpts)
            log.info(f'Successfully load weight from {reload_ckpts_str}, {len(reload_state_dict)} weights are loaded')

        # Give model access to datamodule for post DDP setup processing.
        self._module._datamodule = self._datamodule

    def _setup_dataset(self):

        @rank_zero_only
        def create_synthetic_data_folder(folder_path): 
            os.makedirs(folder_path, exist_ok=False)

        if self._data_cfg.dataset == 'pdb':
            self._train_dataset, self._valid_dataset = eu.dataset_creation(
                PdbDataset, self._cfg.pdb_dataset, self._task, anno=self._cfg.folding.PLM)
            dataset_cfg = self._cfg.pdb_dataset
        else:
            raise ValueError(f'Unrecognized dataset {self._data_cfg.dataset}') 

        return dataset_cfg
        
    def train(self):
        callbacks = []
        logger = WandbLogger(
            **self._exp_cfg.wandb,
        )
        
        # Model checkpoints
        callbacks.append(ModelCheckpoint(**self._exp_cfg.checkpointer))
        
        trainer = Trainer(
            **self._exp_cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=self._train_device_ids,
            plugins=[CustomCheckpointIO()],
        )
        if trainer.is_global_zero:
            ckpt_dir = self._exp_cfg.checkpointer.dirpath
            log.info(f"Checkpoints saved to {ckpt_dir}")
            os.makedirs(ckpt_dir, exist_ok=True)
            cfg_path = os.path.join(ckpt_dir, 'config.yaml')
            with open(cfg_path, 'w') as f:
                OmegaConf.save(config=self._cfg, f=f.name)
            cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
            flat_cfg = dict(eu.flatten_dict(cfg_dict))
            if isinstance(logger.experiment.config, wandb.sdk.wandb_config.Config):
                logger.experiment.config.update(flat_cfg)
        trainer.fit(
            model=self._module,
            datamodule=self._datamodule,
            ckpt_path=self._exp_cfg.warm_start
        )


@hydra.main(version_base=None, config_path="../configs", config_name="base.yaml")
def main(cfg: DictConfig):

    if cfg.experiment.warm_start is not None and cfg.experiment.warm_start_cfg_override:
        # Loads warm start config.
        warm_start_cfg_path = os.path.join(
            os.path.dirname(cfg.experiment.warm_start), 'config.yaml')
        warm_start_cfg = OmegaConf.load(warm_start_cfg_path)

        # Warm start config may not have latest fields in the base config.
        # Add these fields to the warm start config.
        OmegaConf.set_struct(cfg.model, False)
        OmegaConf.set_struct(warm_start_cfg.model, False)
        cfg.model = OmegaConf.merge(cfg.model, warm_start_cfg.model)
        OmegaConf.set_struct(cfg.model, True)
        log.info(f'Loaded warm start config from {warm_start_cfg_path}')

    if cfg.folding.PLM in PLM_models:
        log.info(f'Will use PLM:{cfg.folding.PLM} to encode the sequence')
        
        cfg.folding.PLM = PLM_models[cfg.folding.PLM]
        log.info(f'PLM is loaded from {cfg.folding.PLM}')
    else:
        cfg.folding.PLM = None
        log.info(f'Will not use any PLM in sequence encoding')

    if cfg.pdb_dataset.use_multimer or cfg.pdb_dataset.use_multimer_only:
        cfg.model.node_features.embed_chain = True
        cfg.model.edge_features.embed_chain = True
        cfg.packing_model.node_features.embed_chain = True
        cfg.packing_model.edge_features.embed_chain = True
    
    exp = Experiment(cfg=cfg)
    exp.train()

if __name__ == "__main__":
    main()
