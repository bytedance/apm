"""
Copyright (2025) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0
"""


import os
import sys
from pyrosetta import init, pose_from_pdb, get_fa_scorefxn
from pyrosetta.rosetta.protocols import relax
from pyrosetta.rosetta.core.pack.task import TaskFactory, operation
from pyrosetta.teaching import get_fa_scorefxn
from pyrosetta.rosetta.core.select import residue_selector as selections
from pyrosetta.rosetta.core.select import movemap
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

def relax_and_dG(pdb_file, relax_bb=False, relax_iter=0, relax_repeat=1, save_to=None, dG_config=None):

    random_seed = 1111111

    init(f"-ignore_unrecognized_res false \
            -ignore_zero_occupancy false \
            -load_PDB_components false \
            -relax:default_repeats {relax_repeat} \
            -no_fconfig \
            -mute all", 
            extra_options=f'-constant_seed -jran {random_seed}')
        
    unrelaxed_pose = pose_from_pdb(pdb_file)
    rosetta_fa_scorefunc = get_fa_scorefxn()
    raw_score = rosetta_fa_scorefunc(unrelaxed_pose)
    raw_dG = None

    if not dG_config is None:
        interface_analyzer = InterfaceAnalyzerMover(dG_config, False, rosetta_fa_scorefunc)
        interface_analyzer.set_pack_rounds(0)
        interface_analyzer.set_pack_input(False)
        interface_analyzer.set_compute_packstat(False)
        interface_analyzer.set_pack_separated(False)

        # apply InterfaceAnalyzer
        interface_analyzer.apply(unrelaxed_pose)
        # get dG
        raw_dG = interface_analyzer.get_interface_dG()

    if relax_iter == 0:
        return raw_score, raw_dG, None, None

    # define relaxer
    Relaxer = relax.FastRelax()
    Relaxer.max_iter(relax_iter)

    # define relax range
    N_ligand_res = len(unrelaxed_pose.residues)
    region_index_range = (1, N_ligand_res)
    region_selector = selections.ResidueIndexSelector()
    region_selector.set_index_range(*region_index_range)
    operate_residues = region_selector

    # define movemap to restrict the relax on applied on target region
    relax_mmf = movemap.MoveMapFactory()

    relax_mmf.add_chi_action(movemap.mm_enable, operate_residues)
    if relax_bb:
        relax_mmf.add_bb_action(movemap.mm_enable, operate_residues)
    else:
        relax_mmf.add_bb_action(movemap.mm_disable, operate_residues)

    Relaxer.set_movemap_factory(relax_mmf)

    Relaxer.set_scorefxn(rosetta_fa_scorefunc)
    
    # avoid repacking
    tf = TaskFactory()
    tf.push_back(operation.InitializeFromCommandline())
    tf.push_back(operation.RestrictToRepacking())
    prevent_repacking_rlt = operation.PreventRepackingRLT()
    prevent_subset_repacking = operation.OperateOnResidueSubset(prevent_repacking_rlt, operate_residues, True)
    tf.push_back(prevent_subset_repacking)
    Relaxer.set_task_factory(tf)

    # apply the relaxer
    Relaxer.apply(unrelaxed_pose)
    post_score = rosetta_fa_scorefunc(unrelaxed_pose)

    relaxed_dG = None
    if not dG_config is None:
        # apply InterfaceAnalyzer
        interface_analyzer.apply(unrelaxed_pose)
        # get dG
        relaxed_dG = interface_analyzer.get_interface_dG()

    # save the relaxed pdb file
    if not save_to is None:
        unrelaxed_pose.dump_pdb(save_to)

    return raw_score, raw_dG, post_score, relaxed_dG
    