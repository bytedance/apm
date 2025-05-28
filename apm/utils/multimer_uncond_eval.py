"""
Copyright (2025) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0
"""


import os
import sys
from concurrent.futures import ProcessPoolExecutor
from apm.utils.Rosetta_Relax import relax_and_dG
from apm.utils.arbitrary_RMSD import aligned_RMSD

def APM_multimer_eval_3dG(input_pdb_file, relax_iter=100, relax_repeat=1):
    dG_config = 'A_B'
    sc_relax_pdb = input_pdb_file[:-4] + '_ScRelax.pdb'
    all_relax_pdb = input_pdb_file[:-4] + '_AllRelax.pdb'
    raw_score, raw_dG, sc_relax_score, sc_relax_dG = relax_and_dG(input_pdb_file, relax_bb=False, relax_iter=relax_iter, relax_repeat=relax_repeat, save_to=sc_relax_pdb, dG_config=dG_config)
    _, _, all_relax_score, all_relax_dG = relax_and_dG(input_pdb_file, relax_bb=True, relax_iter=relax_iter, relax_repeat=relax_repeat, save_to=all_relax_pdb, dG_config=dG_config)

    sc_relax_rmsd = aligned_RMSD(input_pdb_file, sc_relax_pdb, RMSD_on='HEAVY', aligned_by='CA', device='cpu')
    all_relax_rmsd = aligned_RMSD(input_pdb_file, all_relax_pdb, RMSD_on='HEAVY', aligned_by='CA', device='cpu')

    return ((raw_dG, sc_relax_dG, all_relax_dG), (sc_relax_rmsd, all_relax_rmsd))

def APM_multimer_eval_2dG(input_pdb_file, relax_iter=100, relax_repeat=1):
    dG_config = 'A_B'
    sc_relax_pdb = input_pdb_file[:-4] + '_ScRelax.pdb'
    all_relax_pdb = input_pdb_file[:-4] + '_AllRelax.pdb'
    _, _, sc_relax_score, sc_relax_dG = relax_and_dG(input_pdb_file, relax_bb=False, relax_iter=relax_iter, relax_repeat=relax_repeat, save_to=sc_relax_pdb, dG_config=dG_config)
    _, _, all_relax_score, all_relax_dG = relax_and_dG(input_pdb_file, relax_bb=True, relax_iter=relax_iter, relax_repeat=relax_repeat, save_to=all_relax_pdb, dG_config=dG_config)

    rmsd = aligned_RMSD(sc_relax_pdb, all_relax_pdb, RMSD_on='CA', aligned_by='CA', device='cpu')

    return sc_relax_dG, all_relax_dG, rmsd

class SuppressStdout:
    def __init__(self, suppress):
        self.suppress = suppress

    def __enter__(self):
        if self.suppress:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.suppress:
            sys.stdout = self._original_stdout

def function_to_apply_eval(args):
    cpu_affinity = list(range(os.cpu_count()))
    os.sched_setaffinity(0, cpu_affinity)

    return args, APM_multimer_eval_2dG(*args)

def apply_multi_process_func(func, args_list, N_cpu=-1):

    if N_cpu == -1:
        N_workers = os.cpu_count() - 1
    else:
        N_workers = min(N_cpu, os.cpu_count()-1)

    with ProcessPoolExecutor(max_workers=N_workers) as executor:
        results = list(executor.map(func, args_list))
    
    return results