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
from pytorch_lightning import Trainer
from omegaconf import DictConfig, OmegaConf
from apm.experiments import utils as eu
from apm.models.flow_module import FlowModule
from apm.data.datasets import PdbDataset
import torch.distributed as dist

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

    def __init__(self, cfg: DictConfig, ckpt_dir):
        """Initialize sampler.

        Args:
            cfg: inference config.
        """

        # Read in checkpoint.
        ckpt_path = cfg.inference.ckpt_path
        ckpt_cfg = OmegaConf.load(os.path.join(ckpt_dir, 'config.yaml'))
        self._original_cfg = cfg.copy()

        # Set-up config.
        OmegaConf.set_struct(cfg, False)
        OmegaConf.set_struct(ckpt_cfg, False)
        cfg = OmegaConf.merge(cfg, ckpt_cfg)
        cfg.experiment.checkpointer.dirpath = './'
        cfg.experiment.rot_inference_style = self._original_cfg.experiment.rot_inference_style
        self._cfg = cfg
        self._cfg.experiment.design_level = self._original_cfg.experiment.design_level
        self._cfg.experiment.training.model_training_steps = 'backbone_1'
        self._exp_cfg = cfg.experiment
        self._infer_cfg = cfg.inference
        self._samples_cfg = self._infer_cfg.samples
        self._rng = np.random.default_rng(self._infer_cfg.seed)

        # Set-up output directory only on rank 0
        local_rank = os.environ.get('LOCAL_RANK', 0)
        if local_rank == 0:
            inference_dir = self.setup_inference_dir()
            self._exp_cfg.inference_dir = inference_dir
            config_path = os.path.join(inference_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                OmegaConf.save(config=self._cfg, f=f)
            log.info(f'Saving inference config to {config_path}')

        # Read checkpoint and initialize module.
        if not ckpt_path is None:
            self._flow_module = FlowModule.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                cfg=self._cfg,
                dataset_cfg=eu.get_dataset_cfg(cfg),
                folding_cfg=self._infer_cfg.folding,
                map_location=torch.device('cuda:0'),
                strict=False
            )
            log.info(f'Successfully load weight from {ckpt_path}')
        else:
            self._flow_module: LightningModule = FlowModule(
                self._cfg,
                eu.get_dataset_cfg(cfg),
                folding_cfg=self._infer_cfg.folding,
            )
        log.info(pl.utilities.model_summary.ModelSummary(self._flow_module))
        self._flow_module.eval()
        self._flow_module._infer_cfg = self._infer_cfg
        self._flow_module._samples_cfg = self._samples_cfg

    @property
    def inference_dir(self):
        return self._flow_module.inference_dir

    def setup_inference_dir(self):
        output_dir = os.path.join(
            self._infer_cfg.predict_dir,
            self._infer_cfg.inference_subdir,
        )
        os.makedirs(output_dir, exist_ok=True)
        log.info(f'Saving results to {output_dir}')
        return output_dir

    def run_sampling(self):
        devices = GPUtil.getAvailable(
            order='memory', limit = 8)[:self._infer_cfg.num_gpus]
        devices = -1
        log.info(f"Using devices: {devices}")
        log.info(f'Evaluating {self._infer_cfg.task}')
        if self._infer_cfg.task == 'unconditional':
            eval_dataset = eu.LengthDataset(self._samples_cfg)
        elif self._infer_cfg.task.startswith('unconditional_multimer'):
            eval_dataset = eu.LengthDataset_multimer(self._samples_cfg)
        elif self._infer_cfg.task.startswith('forward_folding') or self._infer_cfg.task.startswith('inverse_folding'):
            # print(f'***************** using dataset {self._original_cfg.pdb_post2021_dataset}')
            # We want to use the inference settings for the pdb dataset, not what was in the ckpt config
            self._cfg.pdb_post2021_dataset = self._original_cfg.pdb_post2021_dataset
            eval_dataset, _ = eu.dataset_creation(
                PdbDataset, self._cfg.pdb_post2021_dataset, 'hallucination'
            )
        else:
            raise ValueError(f'Unknown task {self._infer_cfg.task}')
        dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=1, shuffle=False, drop_last=False)
        trainer = Trainer(
            accelerator="gpu",
            strategy="ddp",
            devices=devices,
        )
        trainer.predict(self._flow_module, dataloaders=dataloader)

    def compute_unconditional_metrics(self, output_dir):
        log.info(f'Calculating metrics for {output_dir}')
        top_sample_csv = eu.get_all_top_samples(output_dir)
        top_sample_csv['designable'] = top_sample_csv.bb_rmsd <= 2.0
        metrics_df = pd.DataFrame(data={ 
            'Total codesignable': top_sample_csv.designable.sum(),
            'Designable': top_sample_csv.designable.mean(),
            'Average_bb_rmsd': top_sample_csv.bb_rmsd.mean(),
            'Average_bb_tmsc': top_sample_csv.bb_tmsc.mean(),
            'Average_Helix': top_sample_csv.helix_percent.mean(),
            'Average_Strand': top_sample_csv.strand_percent.mean(),
            'Total samples': len(top_sample_csv),
        }, index=[0])
        designable_csv_path = os.path.join(output_dir, 'designable.csv')
        metrics_df.to_csv(designable_csv_path, index=False)
        eu.calculate_diversity(
            output_dir, metrics_df, top_sample_csv, designable_csv_path)

    def compute_forward_folding_metrics(self, output_dir):
        if self._infer_cfg.task == "forward_folding":
            log.info(f'Calculating metrics for {output_dir}')
            top_sample_csv = eu.get_all_top_samples(output_dir)
            top_sample_csv['fold_match_seq'] = top_sample_csv.bb_rmsd_to_gt <= 2.0
            metrics_df = pd.DataFrame(data={ 
                'Total Match Seq': top_sample_csv.fold_match_seq.sum(),
                'Prop Match Seq': top_sample_csv.fold_match_seq.mean(),
                'Average bb_rmsd_to_gt': top_sample_csv.bb_rmsd_to_gt.mean(),
                'Average bb_tmsc_to_gt': top_sample_csv.bb_tmsc_to_gt.mean(),
                'Average fold model bb_rmsd_to_gt': top_sample_csv.fold_model_bb_rmsd_to_gt.mean(),
                'Average fold model bb_tmsc_to_gt': top_sample_csv.fold_model_bb_tmsc_to_gt.mean(),
                'Total samples': len(top_sample_csv),
            }, index=[0])
            metrics_csv_path = os.path.join(output_dir, 'forward_fold_metrics.csv')
            metrics_df.to_csv(metrics_csv_path, index=False)
        elif self._infer_cfg.task == "forward_folding_multimer":
            log.info(f'Calculating metrics for {output_dir}')
            multimer_metric = eu.calculate_multimer_folding_metrics(output_dir)

            metrics_df = pd.DataFrame(data={
                'bb_rmsd': np.mean(multimer_metric['bb_rmsd']),
                'bb_tmsc': np.mean(multimer_metric['bb_tmsc']),
                'chains_rmsd': np.mean(multimer_metric['chains_rmsd']),
                'bb_rmsd_median': np.median(multimer_metric['bb_rmsd']),
                'bb_tmsc_median': np.median(multimer_metric['bb_tmsc']),
                'chains_rmsd_median': np.median(multimer_metric['chains_rmsd']),
                'Total samples': len(multimer_metric['bb_rmsd']),
            }, index=[0])
            metrics_csv_path = os.path.join(output_dir, 'forward_fold_metrics.csv')
            metrics_df.to_csv(metrics_csv_path, index=False)
                
            
    def compute_inverse_folding_metrics(self, output_dir):
        log.info(f'Calculating metrics for {output_dir}')
        top_sample_csv = eu.get_all_top_samples(output_dir)
        top_sample_csv['designable'] = top_sample_csv.bb_rmsd <= 2.0
        metrics_df = pd.DataFrame(data={ 
            'Total designable': top_sample_csv.designable.sum(),
            'Designable': top_sample_csv.designable.mean(),
            'Total samples': len(top_sample_csv),
            'Average_bb_rmsd': top_sample_csv.bb_rmsd.mean(),
            'Average_bb_tmsc': top_sample_csv.bb_tmsc.mean(),
            'Average_seq_recovery': top_sample_csv.inv_fold_seq_recovery.mean(),
            'Average_pmpnn_bb_rmsd': top_sample_csv.pmpnn_bb_rmsd.mean(),
            'Average_pmpnn_seq_recovery': top_sample_csv.pmpnn_seq_recovery.mean(),
        }, index=[0])
        metrics_csv_path = os.path.join(output_dir, 'inverse_fold_metrics.csv')
        metrics_df.to_csv(metrics_csv_path, index=False)



@hydra.main(version_base=None, config_path="../configs", config_name="inference_unconditional")
def run(cfg: DictConfig) -> None:

    # check if ckpt exists
    if not os.path.exists(cfg.inference.ckpt_path):
        raise FileNotFoundError(f'Checkpoint {cfg.inference.ckpt_path} not found')
    else:
        original_ckpt = torch.load(cfg.inference.ckpt_path, map_location=torch.device('cuda:0'))
        ckpt_training_cfg = original_ckpt['hyper_parameters']
        masking_type = 'masking'
        use_multimer = True
        use_PLM = ckpt_training_cfg['folding_cfg']['PLM']
        use_PLM_in_sidechain_model = False
        sidechain_plm_related_weight = [wt for wt in original_ckpt['state_dict'] if wt.startswith("model.sidechain.plm_s")]
        if len(sidechain_plm_related_weight) > 0:
            use_PLM_in_sidechain_model = True
            log.info((f'Following the ckpt, will use PLM:{use_PLM} in sidechain model'))

        ckpt_base_name = os.path.basename(cfg.inference.ckpt_path).split('.')[0] # the name of experiment of which the checkpoint is gained
        ckpt_dir = os.path.dirname(cfg.inference.ckpt_path)
        ckpt_dir_name = os.path.basename(ckpt_dir) # checkpoint id
        
    eval_id = f'{cfg.inference.task}/{ckpt_base_name}' # {exp_name} - {exp_desc} - {eval_task} / {ckpt_id}

    cfg.inference.inference_subdir = eval_id

    cfg.inference.folding.PLM = use_PLM
    log.info(f'Will use PLM : {use_PLM} to encode the sequence')

    if masking_type == 'uniform':
        cfg.inference.interpolant.aatypes.interpolant_type = 'uniform'
        cfg.model.aatype_pred_num_tokens = 20

    if use_multimer:
        cfg.model.node_features.embed_chain = True
        cfg.model.edge_features.embed_chain = True

    cfg.packing_model.use_plm = use_PLM_in_sidechain_model
    # Read model checkpoint.
    log.info(f'Starting inference with {cfg.inference.num_gpus} GPUs')
    start_time = time.time()
    sampler = EvalRunner(cfg, ckpt_dir)
    sampler.run_sampling()
    
    def compute_metrics():
        if cfg.inference.task.startswith('unconditional'):
            sampler.compute_unconditional_metrics(sampler.inference_dir)
        elif cfg.inference.task.startswith('forward_folding'):
            sampler.compute_forward_folding_metrics(sampler.inference_dir)
        elif cfg.inference.task.startswith('inverse_folding'):
            sampler.compute_inverse_folding_metrics(sampler.inference_dir)
        else:
            raise ValueError(f'Unknown task {cfg.inference.task}')

    if dist.is_initialized():
        if dist.get_rank() == 0:
            compute_metrics()
    else:
        compute_metrics()

    elapsed_time = time.time() - start_time
    log.info(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    run()