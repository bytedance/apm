"""
Copyright (2025) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0
"""


import os
import sys
import glob
import time
import shutil
import subprocess

def process_foldseek_aln_file(aln_file, format_output=None):
    aln_res = {}
    last_query = '----'
    with open(aln_file, 'r') as f:
        for l in f:
            l_ = l.strip().split('\t')
            if format_output is None:
                format_output = ['query', 'target'] + [f'term_{i}' for i in range(len(l_)-2)] # the first two term in foldseek alignment file is query and target by default
            curr_query = l_[0]
            if curr_query != last_query:
                aln_res[curr_query] = {term:[] for term in format_output[1:]}
                last_query = curr_query
            for t, v in zip(format_output[1:], l_[1:]):
                if t != 'target':
                    v = float(v)
                aln_res[curr_query][t].append(v)
    return aln_res

def call_foldseek_align(query_pdb, search_db, output_file=None, prefix=None, exhaustive=False):

    # search query_pdb in search_db, and output the result to output_file
    # if output_file is None, the result will be returned

    # query_pdb: str, the path to the query pdb file
    #         or list, contain several pdb files
    #         IT MUST BE NOTICED !!! if the query_pdb is a list, the basename of each pdb file must be different, otherwise the results will not be distinguished
    # search_db: str, the path to the search db file, a pre-built pdb database is /path/to/foldseek_pdb_db
    # output_file: str, the path to the output file

    format_output = ['query', 'target', 'alntmscore', 'qtmscore', 'ttmscore', 'lddt']
    # format_output = ['query', 'target', 'alntmscore', 'lddt']
    format_output_str = ','.join(format_output)

    search_db_dir = os.path.dirname(search_db)
    if prefix is None:
        tmpFolder = os.path.join(search_db_dir, f'foldseek_temp_{time.localtime().tm_hour}{time.localtime().tm_min}{time.localtime().tm_sec}')
    else:
        tmpFolder = os.path.join(search_db_dir, f'foldseek_temp_{prefix}_{time.localtime().tm_hour}{time.localtime().tm_min}{time.localtime().tm_sec}')
    os.makedirs(tmpFolder)

    return_res = False
    if output_file is None:
        if prefix is None:
            output_file = os.path.join(search_db_dir, 'aln.m8')
        else:
            output_file = os.path.join(search_db_dir, f'{prefix}_aln.m8')
        return_res = True
    
    if type(query_pdb) == str:
        query_pdb = [query_pdb, ]
    
    foldseek_args = ['foldseek', 
                     'easy-search', 
                     *query_pdb, 
                     search_db, 
                     output_file, 
                     tmpFolder, 
                     '--format-output', 
                     format_output_str, 
                     '--remove-tmp-files', 
                     'TRUE', 
                     '-v', 
                     '0',  
                     '--alignment-type', 
                     '1',
                     '-s', 
                     '7.5',
                    ]
    if exhaustive:
        foldseek_args = foldseek_args[:-2] + \
                        ['--tmscore-threshold',
                        '0.0',
                        '--exhaustive-search', 
                        '--max-seqs', 
                        '10000000000', ]
    process = subprocess.Popen(foldseek_args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    _ = process.communicate()

    if os.path.exists(tmpFolder):
        shutil.rmtree(tmpFolder)

    if return_res:
        aln_res = process_foldseek_aln_file(output_file, format_output=format_output)
        os.remove(output_file)
        return aln_res

if __name__ == '__main__':

    ### python foldseek_align.py query_pdb_folder alignment_output_file
    ### query_pdb_folder : the folder contains all query pdb files
    ### alignment_output_file : the file to save the alignment results

    foldseek_db = 'foldseek_pdb_db'
    query_pdb_folder, alignment_output_file = sys.argv[1:]
    query_pdbs = glob.glob(os.path.join(query_pdb_folder, '*.pdb'))
    call_foldseek_align(query_pdbs, foldseek_db, output_file=alignment_output_file)