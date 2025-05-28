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
import time
import numpy as np
import hydra
import torch
import pandas as pd
import glob
import GPUtil
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf
from apm.experiments import utils as eu
from apm.models.flow_module import FlowModule
from apm.data.conditional_dataset import ConditionalDataset, clean_pdb
import torch.distributed as dist


torch.set_float32_matmul_precision('high')
log = eu.get_pylogger(__name__)


def eight(n):
    return int(8*round(n/8))

def exp2(n):
    return int(np.exp2(round(np.log2(n))))

torch.set_float32_matmul_precision('high')
log = eu.get_pylogger(__name__)

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

PLM_models_inverse = {p:k for k,p in PLM_models.items()}


class EvalRunner:

    def __init__(self, cfg: DictConfig):
        """Initialize sampler.

        Args:
            cfg: inference config.
        """

        # Read in checkpoint.
        ckpt_path = cfg.inference.conditional_ckpt_path
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_cfg = OmegaConf.load(os.path.join(ckpt_dir, 'config.yaml'))
        self._original_cfg = cfg.copy()

        # Set-up config.
        OmegaConf.set_struct(cfg, False)
        OmegaConf.set_struct(ckpt_cfg, False)
        cfg = OmegaConf.merge(cfg, ckpt_cfg)

        cfg.experiment.checkpointer.dirpath = './'
        self._cfg = cfg
        self._exp_cfg = cfg.experiment
        self._infer_cfg = cfg.inference
        self._rng = np.random.default_rng(self._infer_cfg.seed)

        # Set-up output directory only on rank 0
        local_rank = os.environ.get('LOCAL_RANK', 0)
        if local_rank == 0:
            inference_dir = self.setup_inference_dir(ckpt_path)
            self._exp_cfg.inference_dir = inference_dir
            config_path = os.path.join(inference_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                OmegaConf.save(config=self._cfg, f=f)
            log.info(f'Saving inference config to {config_path}')

        # Read checkpoint and initialize module.
        self._infer_cfg.folding.PLM = self._cfg.folding.PLM
        self._flow_module = FlowModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            cfg=self._cfg,
            dataset_cfg=self._cfg.conditional_dataset,
            folding_cfg=self._infer_cfg.folding,
            strict=False,
        )
        log.info(pl.utilities.model_summary.ModelSummary(self._flow_module))
        self._flow_module.eval()
        self._flow_module._infer_cfg = self._infer_cfg

    @property
    def inference_dir(self):
        return self._flow_module.inference_dir

    def setup_inference_dir(self, ckpt_path):
        self._ckpt_name = os.path.basename(ckpt_path).split(".")[0]
        output_dir = os.path.join(
            self._infer_cfg.predict_dir,
            self._infer_cfg.task,
            self._ckpt_name,
            self._infer_cfg.inference_subdir,
            str(self._infer_cfg.seed)
        )
        os.makedirs(output_dir, exist_ok=True)
        log.info(f'Saving results to {output_dir}')
        return output_dir

    def run_sampling(self):
        log.info(f'Evaluating {self._infer_cfg.task}')
        
        eval_dataset = ConditionalDataset(dataset_cfg=self._cfg.conditional_dataset)
        dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=self._infer_cfg.batch_size, shuffle=False, drop_last=False)
        trainer = Trainer(
            accelerator="gpu",
            strategy="ddp",
        )
        self._flow_module._eval_dataset = eval_dataset
        trainer.predict(self._flow_module, dataloaders=dataloader)


@hydra.main(version_base=None, config_path="../configs", config_name="inference_conditional")
def run(cfg: DictConfig) -> None:

    use_PLM_in_sidechain_model = True
    cfg.packing_model.use_plm = use_PLM_in_sidechain_model

    # Read model checkpoint.
    start_time = time.time()
    sampler = EvalRunner(cfg)
    sampler.run_sampling()

    elapsed_time = time.time() - start_time
    log.info(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    run()
