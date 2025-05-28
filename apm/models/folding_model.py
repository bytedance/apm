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
import esm
import subprocess
import logging
import torch
import torch.nn as nn
import json
import numpy as np
from apm.data import utils as du
from biotite.sequence.io import fasta
import pandas as pd
import glob

# from byprot.models.lm.dplm import DiffusionProteinLanguageModel
# from byprot.datamodules.dataset.uniref import DPLMCollater
from transformers import AutoModel

from faesm.esm import FAEsmForMaskedLM
from faesm.esmc import ESMC

PLM_LIB = {
    'esm2_t6_8M_UR50D.pt' : {'repr_layer':6, 'emb_dim': 320}, 
    'esm2_t12_35M_UR50D.pt' : {'repr_layer':12, 'emb_dim': 480},  
    'esm2_t30_150M_UR50D.pt' : {'repr_layer':30, 'emb_dim': 640}, 
    'esm2_t33_650M_UR50D.pt' : {'repr_layer':33, 'emb_dim': 1280}, 
    'esm2_t36_3B_UR50D.pt' : {'repr_layer':36, 'emb_dim': 2560}, 
    'faESMC_600M': {'repr_layer':36, 'emb_dim': 1152},
    'faESM2_650M' : {'repr_layer':33, 'emb_dim': 1280},  
    'dplm_150m': {'repr_layer':30, 'emb_dim': 640},
    'dplm_650m': {'repr_layer':33, 'emb_dim': 1280},
    'gLM2_150M': {'repr_layer':29, 'emb_dim': 640}, # actually 30 trunk layers, but not output embedding layer
    'gLM2_650M': {'repr_layer':32, 'emb_dim': 1280}, # actually 33 trunk layers, but not output embedding layer
}

class FoldingModel(nn.Module):

    def __init__(self, cfg, device_id=None):
        super(FoldingModel, self).__init__()
        self._print_logger = logging.getLogger(__name__)
        self._cfg = cfg
        self._esmf = None
        self._plm = None
        self._device_id = device_id
        self._device = None
        if self._cfg.PLM is not None:
            # self.plm_representations_layer = PLM_LIB[os.path.basename(self._cfg.PLM)]['repr_layer']
            # self.plm_representations_dim = PLM_LIB[os.path.basename(self._cfg.PLM)]['emb_dim']
            self.plm_representations_layer = PLM_LIB[self._cfg.PLM]['repr_layer']
            self.plm_representations_dim = PLM_LIB[self._cfg.PLM]['emb_dim']
        self.PLM_inited = False

    @property
    def device_id(self):
        if self._device_id is None:
            self._device_id = torch.cuda.current_device()
        return self._device_id

    @property
    def device(self):
        if self._device is None:
            self._device = f'cuda:{self.device_id}'
        return self._device
    
    def to_cpu(self):
        if not self._esmf is None:
            self._esmf = self._esmf.to('cpu')
    
    def to_cuda(self):
        if not self._esmf is None:
            self._esmf = self._esmf.to(self.device)

    def fold_fasta(self, fasta_path, output_dir):
        if self._cfg.folding_model == 'esmf':
            folded_output = self._esmf_model(fasta_path, output_dir)
        elif self._cfg.folding_model == 'af2':
            folded_output = self._af2_model(fasta_path, output_dir)
        else:
            raise ValueError(f'Unknown folding model: {self._cfg.folding_model}')
        return folded_output

    @torch.no_grad()
    def _esmf_model(self, fasta_path, output_dir):
        if self._esmf is None:
            self._print_logger.info(f'Loading ESMFold from {self._cfg.pt_hub_dir} on device {self.device}')
            torch.hub.set_dir(self._cfg.pt_hub_dir)
            self._esmf = esm.pretrained.esmfold_v1().eval().to(self.device)
        fasta_seqs = fasta.FastaFile.read(fasta_path)
        folded_outputs = {
            'folded_path': [],
            'header': [],
            'plddt': [],
            'seq': []
        }
        for header, string in fasta_seqs.items():
            # Run ESMFold
            # Need to convert unknown amino acids to alanine since ESMFold 
            # doesn't like them and will remove them...
            string = string.replace('X', 'A')
            esmf_sample_path = os.path.join(output_dir, f'folded_{header}.pdb')
            esmf_outputs = self._esmf.infer(string)
            pdb_output = self._esmf.output_to_pdb(esmf_outputs)[0]
            with open(esmf_sample_path, "w") as f:
                f.write(pdb_output)
            mean_plddt = esmf_outputs['mean_plddt'][0].item()
            folded_outputs['folded_path'].append(esmf_sample_path)
            folded_outputs['header'].append(header)
            folded_outputs['plddt'].append(mean_plddt)
            folded_outputs['seq'].append(string)
        return pd.DataFrame(folded_outputs)

    def get_esm_tk_mapping(self):
        esm_alphabet_list = ['<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']
        multiflow_alphabet_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '<mask>']
        self.esm_tk_mapping = torch.Tensor([1, ] + [
            esm_alphabet_list.index(tk) for tk in multiflow_alphabet_list
        ]).long()
        self.esm_tk_mapping_ = torch.Tensor([esm_alphabet_list.index(tk) for tk in multiflow_alphabet_list]).long()
        ### self.esm_tk_mapping and self.esm_tk_mapping_ are not the same, self.esm_tk_mapping contains a token for <pad>

    def get_plm_emb_weight(self):
        if self.plm_type == 'ESM':
            raw_plm_emb_weight = self._plm.embed_tokens.weight # [33, esm_dim]
        elif self.plm_type == 'DPLM':
            raw_plm_emb_weight = self._plm.net.esm.embeddings.word_embeddings.weight
        self._plm_emb_weight = raw_plm_emb_weight[self.esm_tk_mapping_] # [21, esm_dim]
        self._plm_emb_weight.detach()

    @property
    def plm_emb_weight(self):
        if self._cfg.PLM is None:
            return None
        if self._plm is None:
            self.init_plm()
        return self._plm_emb_weight

    def openfold_tk_2_esm_tk(self, openfold_tk, pad_mask):
        self.esm_tk_mapping = self.esm_tk_mapping.to(openfold_tk.device)
        if not pad_mask is None:
            openfold_tk = (openfold_tk + 1).masked_fill(pad_mask != 1, 0) # pad_mask : 1 for valid residue, 0 for padding
        esm_tk = self.esm_tk_mapping[openfold_tk]

        batch_size = esm_tk.size(0)
        batch_bos = esm_tk.new_full((batch_size, 1), self.plm_token['bosi'])
        batch_eos = esm_tk.new_full((batch_size, 1), self.plm_token['padi'])
        esm_tk = torch.cat([batch_bos, esm_tk, batch_eos], dim=1)
        # Use the first padding index as eos during inference.
        esm_tk[range(batch_size), (esm_tk != 1).sum(1)] = self.plm_token['eosi'] # (esm_tk != 1).sum(1) means the first padding index
        return esm_tk

    def _plm_embedding(self, aatypes, pad_mask=None):
        # get the representations from the last layer of the PLM
        if self._plm is None:
            self.init_plm()
        if pad_mask is None:
            pad_mask = torch.ones_like(aatypes)
        aatypes_esm_style = self.openfold_tk_2_esm_tk(aatypes, pad_mask)
        if self.plm_type == 'ESM':
            with torch.no_grad():
                esm_results = self._plm(aatypes_esm_style, repr_layers=[self.esm_representations_layer], return_contacts=True)
            token_representations = esm_results["representations"][self.esm_representations_layer]
            token_representations = token_representations[:, 1:-1] # eos representations could be preserved as residue_mask will ignore the non-residue tokens, and eos and pad have no difference in subsequent calculations
            return token_representations
        elif self.plm_type == 'DPLM':
            with torch.no_grad():
                dplm_results = self._plm.net(aatypes_esm_style)
            token_representations = dplm_results['last_hidden_state']
            token_representations = token_representations[:, 1:-1] 
            return token_representations

    def _plm_embedding_all_layer(self, aatypes, pad_mask=None, attn_mask=None, get_attn_map=False):
        # get the representations from the all the layers of the PLM (including the embedding layer)
        if self._plm is None:
            self.init_plm()
        # if pad_mask is None:
        #     pad_mask = torch.ones_like(aatypes)
        aatypes_esm_style = self.openfold_tk_2_esm_tk(aatypes, pad_mask)
        if self.plm_type == 'ESM':
            with torch.no_grad():
                esm_results = self._plm(aatypes_esm_style, repr_layers=range(self.plm_representations_layer+1), return_contacts=False, attn_mask=attn_mask)
                token_representations = [esm_results["representations"][li][:,1:-1,:] for li in range(self.plm_representations_layer+1)]
                token_representations = torch.stack(token_representations, dim=2) # [B, L, num_layer+1, H]
                if get_attn_map:
                    esm_attns = esm_results["attentions"].permute(0, 4, 3, 1, 2).flatten(3, 4)[:, 1:-1, 1:-1, :] # [B, L, L, num_layer*num_head]
                else:
                    esm_attns = None
            return token_representations, esm_attns
        elif self.plm_type == 'DPLM':
            with torch.no_grad():
                dplm_results = self._plm.net(aatypes_esm_style)
                token_representations = [dplm_results["hidden_states"][li][:,1:-1,:] for li in range(self.plm_representations_layer+1)]
                token_representations = torch.stack(token_representations, dim=2) # [B, L, num_layer+1, H]
                if get_attn_map:
                    raise NotImplementedError("We do not implement get_attn_map for DPLM.")
                dplm_attns = None
            return token_representations, dplm_attns
        
    def _esm_tfmr(self, node_emb, src_key_padding_mask=None):
        if self._esm is None:
            self.init_esm()
        # part of ESM forward()
        node_emb = node_emb.transpose(0, 1)
        for layer_idx, layer in enumerate(self._esm.layers):
            node_emb, attn = layer(
                node_emb,
                self_attn_padding_mask=None,
                need_head_weights=False,
            )
        
        node_emb = self._esm.emb_layer_norm_after(node_emb)
        node_emb = node_emb.transpose(0, 1)
        return node_emb

    def _esmf_esm_embedding(self, aatypes, residue_mask):
        if self._esmf is None:
            self._print_logger.info(f'Loading ESMFold from {self._cfg.pt_hub_dir} on device {self.device}')
            torch.hub.set_dir(self._cfg.pt_hub_dir)
            self._esmf = esm.pretrained.esmfold_v1().eval().to(self.device)
        esmaa = self._esmf._af2_idx_to_esm_idx(aatypes, residue_mask).to(self.device)
        esm_s = self._esmf._compute_language_model_representations(esmaa)
        return esm_s[:,:,-1,:].type(torch.float32).detach()

    def _esmf_esm_tfmr(self, node_emb, src_key_padding_mask=None):
        if self._esmf is None:
            self._print_logger.info(f'Loading ESMFold from {self._cfg.pt_hub_dir} on device {self.device}')
            torch.hub.set_dir(self._cfg.pt_hub_dir)
            self._esmf = esm.pretrained.esmfold_v1().eval().to(self.device)
        node_emb = node_emb.type(torch.float16)
        node_emb = node_emb.transpose(0, 1)
        for layer_idx, layer in enumerate(self._esmf.esm.layers[-8:]):
            node_emb, attn = layer(
                node_emb,
                self_attn_padding_mask=None,
                need_head_weights=False,
            )
        
        node_emb = self._esmf.esm.emb_layer_norm_after(node_emb)
        node_emb = node_emb.transpose(0, 1)
        node_emb = node_emb.type(torch.float32)

    def init_gLM(self, ):
        self._plm = AutoModel.from_pretrained(f'tattabio/{self._cfg.PLM}', trust_remote_code=True).to(self.device)
        self._plm.requires_grad_(False)
        self.gLM_tk_mapping = torch.Tensor([5, 10, 17, 13, 23, 16, 9, 6, 21, 12, 4, 15, 20, 18, 14, 8, 11, 22, 19, 7, 35, 33]).long().to(self.device)
        # gLM2 vocab:
        # {'X': 24, 'I': 12, 'Q': 16, 'P': 14, 'Z': 27, 'M': 20, 'S': 8, 'c': 31, 'G': 6, 't': 30, '<sep>': 36, 
        #  '<pad>': 1, 'E': 9, 'W': 22, '<unk>': 3, '<eos>': 2, '<mask>': 35, 'O': 28, 'Y': 19, 'V': 7, 'U': 26, 
        #  'H': 21, 'F': 18, 'L': 4, '<cls>': 0, 'C': 23, 'R': 10, '<+>': 33, 'D': 13, 'K': 15, 'N': 17, 'T': 11, 
        #  'B': 25, '<->': 34, 'g': 32, 'a': 29, 'A': 5}
        self._plm_emb_weight = self._plm.tok_embeddings.weight[self.gLM_tk_mapping][:-1].clone() # the last token, <+> is removed, to maintain the shape of [21, hidden_dim]
        self._plm_emb_weight.detach()
        self.PLM_inited = True
    
    def gLM_encoding(self, aatypes_in_gLM_temmplates):
        if self._plm is None:
            self.init_gLM()
        aatype_in_gLM_style = self.gLM_tk_mapping[aatypes_in_gLM_temmplates]
        with torch.no_grad():
            gLM_hidden_states = self._plm(aatype_in_gLM_style.long(), output_hidden_states=True).hidden_states
            gLM_hidden_reprs = torch.stack(gLM_hidden_states, dim=2) # [B, L, num_layer, H], !!! NOTICE !!!, gLM2 does not return the embedding layer output
        return gLM_hidden_reprs
    
    def init_faESM(self, ):
        if self._cfg.PLM == 'faESMC_600M':
            self._plm = ESMC.from_pretrained("esmc_600m", use_flash_attn=True).to(self.device).eval().to(torch.bfloat16)
            _plm_emb_weight_raw = self._plm.embed.weight.float()
            # ESMC vocab:
            # {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 
            #  'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 
            #  'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, 
            #  '<null_1>': 31, '<mask>': 32}
        elif self._cfg.PLM == 'faESM2_650M':
            self._plm = FAEsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D").to(self.device).eval().to(torch.bfloat16)
            _plm_emb_weight_raw = self._plm.esm.embeddings.word_embeddings.weight.float()
            # ESM2 vocab:
            # {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 
            #  'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 
            #  'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, 
            #  '<null_1>': 31, '<mask>': 32}
        else:
            raise NotImplementedError(f'PLM {self.PLM} is not supported.')
        
        self._plm.requires_grad_(False)
        self.tk_mapping = torch.Tensor([5, 10, 17, 13, 23, 16, 9, 6, 21, 12, 4, 15, 20, 18, 14, 8, 11, 22, 19, 7, 32, 0, 2]).long().to(self.device) # ESMC and ESM2 have the same vocab
        self._plm_emb_weight = _plm_emb_weight_raw[self.tk_mapping][:-2].clone().detach() # the last two tokens, <cls> and <eos>, are removed, to maintain the shape of [21, hidden_dim]
        # self._plm_emb_weight.detach()
        self.PLM_inited = True
    
    def faESM2_encoding(self, aatypes_in_ESM_templates, cu_seqlens, max_seqlen):
        if not self.PLM_inited:
            self.init_faESM()
        aatype_in_ESM_style = self.tk_mapping[aatypes_in_ESM_templates]
        with torch.no_grad():
            plm_output = self._plm.customized_forward(input_ids=aatype_in_ESM_style, 
                                                      attention_mask=torch.ones_like(aatype_in_ESM_style), 
                                                      output_hidden_states=True, 
                                                      cu_seqlens=cu_seqlens, 
                                                      max_seqlen=max_seqlen)
            plm_hidden_reprs = torch.stack(plm_output['hidden_states'], dim=-2).squeeze(0) # [L, num_layer, H], samples within one batch is concatenated together
        return plm_hidden_reprs

    def faESMC_encoding(self, aatypes_in_ESM_templates, cu_seqlens, max_seqlen):
        if not self.PLM_inited:
            self.init_faESM()
        aatype_in_ESM_style = self.tk_mapping[aatypes_in_ESM_templates]
        with torch.no_grad():
            plm_output = self._plm.customized_forward(sequence_tokens=aatype_in_ESM_style, 
                                                      cu_seqlens = cu_seqlens,
                                                      max_seqlen=max_seqlen,
                                                      return_attn_maps=False)
            plm_hidden_reprs = torch.stack(plm_output.embeddings, dim=-2) # [L, num_layer, H], samples within one batch is concatenated together
        return plm_hidden_reprs

    def _af2_model(self, fasta_path, output_dir):
        af2_args = [
            self._cfg.colabfold_path,
            fasta_path,
            output_dir,
            '--msa-mode',
            'single_sequence',
            '--num-models',
            '1',
            '--random-seed',
            '123',
            '--device',
            f'{self.device_id}',
            '--model-order',
            '4',
            '--num-recycle',
            '3',
            '--model-type',
            'alphafold2_ptm',
        ]
        process = subprocess.Popen(
            af2_args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
        _ = process.wait()
        fasta_seqs = fasta.FastaFile.read(fasta_path)
        folded_outputs = {
            'folded_path': [],
            'header': [],
            'plddt': [],
        }
        all_af2_files = glob.glob(os.path.join(output_dir, '*'))
        af2_model_4_pdbs = {}
        af2_model_4_jsons = {}
        for x in all_af2_files:
            if 'model_4' in x:
                seq_name = os.path.basename(x)
                if x.endswith('.json'):
                    seq_name = seq_name.split('_scores')[0]
                    af2_model_4_jsons[seq_name] = x
                if x.endswith('.pdb'):
                    seq_name = seq_name.split('_unrelaxed')[0]
                    af2_model_4_pdbs[seq_name] = x
            else:
                os.remove(x)
        for header, _ in fasta_seqs.items():
            af2_folded_path = af2_model_4_pdbs[header]
            af2_json_path = af2_model_4_jsons[header]
            with open(af2_json_path, 'r') as f:
                folded_confidence = json.load(f)
            mean_plddt = np.mean(folded_confidence['plddt'])
            folded_outputs['folded_path'].append(af2_folded_path)
            folded_outputs['header'].append(header)
            folded_outputs['plddt'].append(mean_plddt)
        return pd.DataFrame(folded_outputs)

    def run_pmpnn(self, input_dir, output_path):

        os.makedirs(os.path.join(input_dir, 'seqs'), exist_ok=True)
        process = subprocess.Popen([
            'python',
            os.path.join(self._cfg.pmpnn_path,
                         'helper_scripts/parse_multiple_chains.py'),
            f'--input_path={input_dir}',
            f'--output_path={output_path}',
        ])
        _ = process.wait()

        pmpnn_args = [
            'python',
            os.path.join(self._cfg.pmpnn_path, 'protein_mpnn_run.py'),
            '--out_folder',
            input_dir,
            '--jsonl_path',
            output_path,
            '--num_seq_per_target',
            str(self._cfg.seq_per_sample),
            '--sampling_temp',
            '0.1',
            '--seed',
            '38',
            '--batch_size',
            '1',
            '--device',
            str(self.device_id),
        ]
        process = subprocess.Popen(
            pmpnn_args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
        _ = process.wait()

    def run_pmpnn_multimer(self, input_dir, output_path):

        os.makedirs(os.path.join(input_dir, 'seqs'), exist_ok=True)
        process = subprocess.Popen([
            'python',
            os.path.join(self._cfg.pmpnn_path,
                         'helper_scripts/parse_multiple_chains.py'),
            f'--input_path={input_dir}',
            f'--output_path={output_path}',
        ])
        _ = process.wait()

        # read the parsed pdb file to get chain id
        with open(output_path, 'r') as jf:
            jl = jf.readlines()[0] # only one pdb in this file
            pdb_info_dict = json.loads(jl)
        
        chain_ids = [t.split('_')[-1] for t in pdb_info_dict.keys() if t.startswith('seq_chain')]
        chains_to_design=" ".join(chain_ids)

        path_for_assigned_chains = output_path[:-4] + '_assigned_pdbs.jsonl'
        process = subprocess.Popen([
            'python',
            os.path.join(self._cfg.pmpnn_path,
                         'helper_scripts/assign_fixed_chains.py'),
            f'--input_path={output_path}',
            f'--output_path={path_for_assigned_chains}',
            '--output_path',
            path_for_assigned_chains,
            '--chain_list',
            chains_to_design,

        ])
        _ = process.wait()

        pmpnn_args = [
            'python',
            os.path.join(self._cfg.pmpnn_path, 'protein_mpnn_run.py'),
            '--out_folder',
            input_dir,
            '--jsonl_path',
            output_path,
            '--chain_id_jsonl', 
            path_for_assigned_chains,
            '--num_seq_per_target',
            str(self._cfg.seq_per_sample),
            '--sampling_temp',
            '0.1',
            '--seed',
            '38',
            '--batch_size',
            '1',
            '--device',
            str(self.device_id),
        ]
        process = subprocess.Popen(
            pmpnn_args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
        _ = process.wait()

    def run_pmpnn_condition_seq(self, input_dir, output_path, fixed_index):

        os.makedirs(os.path.join(input_dir, 'seqs'), exist_ok=True)
        
        # parse the input pdb file
        process = subprocess.Popen([
            'python',
            os.path.join(self._cfg.pmpnn_path,
                         'helper_scripts/parse_multiple_chains.py'),
            f'--input_path={input_dir}',
            f'--output_path={output_path}',
        ])
        _ = process.wait()

        # read the parsed pdb file to get chain id
        with open(output_path, 'r') as jf:
            jl = jf.readlines()[0] # only one pdb in this file
            pdb_info_dict = json.loads(jl)
        
        chain_ids = [t.split('_')[-1] for t in pdb_info_dict.keys() if t.startswith('seq_chain')]
        chain_id = chain_ids[0] # only one chain in this pdb
        chains_to_design=chain_id
        
        # parse chain id info
        path_for_assigned_chains = output_path[:-4] + '_assigned_pdbs.jsonl'
        assign_args = [
            'python',
            os.path.join(self._cfg.pmpnn_path,
                         'helper_scripts/assign_fixed_chains.py'),
            f'--input_path={output_path}',
            f'--output_path={path_for_assigned_chains}',
            f'--chain_list', 
            chains_to_design,
        ]
        process = subprocess.Popen(
            assign_args,
        )
        _ = process.wait()

        # parse fix residues info
        motif_pos = [str(i+1) for i in fixed_index]
        fixed_positions = ' '.join(motif_pos)
        path_for_fixed_positions = output_path[:-4] + '_fixed_pdbs.jsonl'
        fixed_args = [
            'python',
            os.path.join(self._cfg.pmpnn_path,
                         'helper_scripts/make_fixed_positions_dict.py'),
            f'--input_path={output_path}',
            f'--output_path={path_for_fixed_positions}',
            f'--chain_list', 
            chains_to_design,
            f'--position_list', 
            fixed_positions,
        ]
        process = subprocess.Popen(
            fixed_args,
        )
        _ = process.wait()

        # run ProteinMPNN
        pmpnn_args = [
            'python',
            os.path.join(self._cfg.pmpnn_path, 'protein_mpnn_run.py'),
            '--out_folder',
            input_dir,
            '--jsonl_path',
            output_path,
            '--chain_id_jsonl', 
            path_for_assigned_chains, 
            '--fixed_positions_jsonl', 
            path_for_fixed_positions, 
            '--num_seq_per_target',
            str(self._cfg.seq_per_sample),
            '--sampling_temp',
            '0.1',
            '--seed',
            '38',
            '--batch_size',
            '1',
            '--device',
            str(self.device_id),
        ]
        process = subprocess.Popen(
            pmpnn_args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
        _ = process.wait()
