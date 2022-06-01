#!/usr/bin/env python
import treetime
import numpy as np
import os,sys
import datetime
import subprocess
import re
import click
from Bio import Phylo

import utility_functions_flu as flu_utils
import utility_functions_general as gen_utils
from utility_functions_beast import run_beast, read_beast_log


def _run_beast(N_leaves, subtree_filename, out_dir, res_file, aln_name, template_file):

    def beast_log_post_process(log_file):
        df = read_beast_log(log_file, np.max(dates.values()))
        if df is None or df.shape[0] < 200:
            print ("Beast log {} is corrupted or BEAST run did not finish".format(log_file))
            return
        inferred_LH = df['likelihood'][-50:].mean()
        inferred_LH_std = df['likelihood'][-50:].std()
        inferred_Tmrca = df['treeModel.rootHeight'][-50:].mean()
        inferred_Tmrca_std = df['treeModel.rootHeight'][-50:].std()
        inferred_Mu = df['clock.rate'][-50:].mean()
        inferred_Mu_std = df['clock.rate'][-50:].std()

        if not os.path.exists(res_file):
            try:
                with open(res_file, 'w') as of:
                    of.write("#Filename,N_leaves,LH,LH_std,Tmrca,Tmrca_std,Mu,Mu_std\n")
            except:
                pass

        with open(res_file, 'a') as of:
            of.write("{},{},{},{},{},{},{},{}\n".format(
                subtree_filename,
                N_leaves,
                inferred_LH,
                inferred_LH_std,
                inferred_Tmrca,
                inferred_Tmrca_std,
                inferred_Mu,
                inferred_Mu_std))

    dates = flu_utils.dates_from_flu_tree(subtree_filename)
    beast_out_dir = os.path.join(out_dir, 'beast_out')
    if not os.path.exists(beast_out_dir):
        try:
            os.makedirs(beast_out_dir)
        except:
            pass
    beast_prefix = os.path.join(beast_out_dir, os.path.split(subtree_filename)[-1][:-4])  # truncate '.nwk'
    run_beast(subtree_filename, aln_name, dates, beast_prefix,
    template_file=template_file,
    log_post_process=beast_log_post_process)

def sample_subtree(out_dir, N_leaves, subtree_fname_suffix, tree_name, aln_name):
    subtrees_dir = os.path.join(out_dir, "subtrees")
    if not os.path.exists(subtrees_dir):
        try:
            os.makedirs(subtrees_dir)
        except:
            pass
    subtree_fname_format = "H3N2_HA_2011_2013_{}_{}.nwk".format(N_leaves, subtree_fname_suffix)
    subtree_filename = os.path.join(subtrees_dir, subtree_fname_format)
    print(tree_name, N_leaves, subtree_filename, aln_name)
    tree = flu_utils.subtree_with_same_root(tree_name, N_leaves, subtree_filename, aln_name)
    N_leaves = tree.count_terminals()
    return subtree_filename, N_leaves

@click.command()
@click.option('--size', required=True, type=int, help='number of leaves')
@click.option('--out_dir', type=click.UNPROCESSED, required=True, help='working directory')
@click.option('--suffix', type=click.UNPROCESSED, required=True, help='number of leaves')
@click.option('--treetime_file', type=click.UNPROCESSED, default=None, help='treetime output file')
@click.option('--lsd_file', type=click.UNPROCESSED, default=None, help='lsd output file')
@click.option('--beast_file', type=click.UNPROCESSED, default=None, help='beast output file')
@click.option('--aln_file', required=True, type=click.UNPROCESSED, help='alignment file')
@click.option('--tree_file', required=True, type=click.UNPROCESSED, help='tree file')
@click.option('--template_file', required=True, type=click.UNPROCESSED, help='beast template file')
@click.option('--lsd_params', help='additional lsd parameters')
def run(size, out_dir, suffix, treetime_file, lsd_file, beast_file, aln_file, tree_file, template_file, lsd_params):
    if lsd_params is not None:
        lsd_params = lsd_params.split("|")
    else:
        lsd_params = ['-c', '-r', 'a', '-v']



    #  Sample subtree
    subtree_filename, size = sample_subtree(out_dir, size, suffix, tree_file, aln_file)

    if treetime_file is not None:
        treetime_outdir = os.path.join(out_dir, 'treetime_out')
        try:
            os.makedirs(treetime_outdir)
        except:
            pass

        dates = flu_utils.dates_from_flu_tree(tree_file)
        myTree = treetime.TreeTime(gtr='Jukes-Cantor',
            tree=subtree_filename, aln=aln_file, dates=dates,
            debug=False, verbose=4)
        myTree.optimize_seq_and_branch_len(reuse_branch_len=True, prune_short=True, max_iter=5, infer_gtr=False)
        start = datetime.datetime.now()
        myTree.run(root='best', relaxed_clock=False, max_iter=3, resolve_polytomies=True, do_marginal=False)
        end = datetime.datetime.now()

        treetime_outfile = os.path.join(treetime_outdir, os.path.split(subtree_filename)[-1].replace(".nwk", ".tree"))
        Phylo.write(myTree.tree, treetime_outfile, 'nexus')

        if not os.path.exists(treetime_file):
            try:
                with open(treetime_file, 'w') as of:
                    of.write("#Filename,N_leaves,Tmrca,Mu,R^2(initial clock),R^2(internal nodes),Runtime\n")
            except:
                pass

        with open(treetime_file, 'a') as of:
            of.write("{},{},{},{},{},{},{}\n".format(
                subtree_filename,
                str(size),
                str(myTree.tree.root.numdate),
                str(myTree.date2dist.clock_rate),
                str(myTree.date2dist.r_val),
                str(gen_utils.internal_regress(myTree)),
                str((end-start).total_seconds())    ))
        print ("TreeTime done!")
    else:
        print ("Skip TreeTime run")


    if lsd_file is not None:
        lsd_outdir = os.path.join(out_dir, 'LSD_out')
        #  run LSD for the subtree:
        if not os.path.exists(lsd_outdir):
            try:
                os.makedirs(lsd_outdir)
            except:
                pass
        lsd_outfile = os.path.join(lsd_outdir, os.path.split(subtree_filename)[-1].replace(".nwk", ".txt"))
        datesfile = os.path.join(lsd_outdir, os.path.split(subtree_filename)[-1].replace(".nwk", ".lsd_dates.txt"))
        flu_utils.create_LSD_dates_file_from_flu_tree(subtree_filename, datesfile)
        runtime = gen_utils.run_LSD(subtree_filename, datesfile, lsd_outfile, lsd_params)
        #  parse LSD results
        tmrca, mu, objective = gen_utils.parse_lsd_output(lsd_outfile)
        try:
            if float(mu) > 0:

                if not os.path.exists(lsd_file):
                    try:
                        with open(lsd_file, 'w') as of:
                            of.write("#Filename,N_leaves,Tmrca,Mu,Runtime,Objective\n")
                    except:
                        pass

                with open(lsd_file, "a") as of:
                    of.write(",".join([subtree_filename, str(size), tmrca, mu, runtime, objective]))
                    of.write("\n")
        except:
            pass

        print ("LSD Done!")
    else:
        print ("Skip LSD run")

    if beast_file is not None:
        _run_beast(size, subtree_filename, out_dir, beast_file, aln, template_file)

if __name__ == "__main__":
    run()






