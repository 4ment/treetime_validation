#!/usr/bin/env python
import subprocess as sp
import os

CLUSTER = True

if __name__ =="__main__":


    #output directories
    out_dir = "./flu_H3N2/missing_dates/"
    subtree_dir = os.path.join(out_dir, "subtrees")

    # without extension
    subtree_files = ["./flu_H3N2/missing_dates/subtrees/H3N2_HA_2011_2013_100seqs"]
    dates_knonwn_fraction = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    Npoints = 10

    #subtree_files = ["./flu_H3N2/missing_dates/subtrees/H3N2_HA_2011_2013_100seqs"]
    #dates_knonwn_fraction = [0.5]
    #Npoints = 1

    for subtree in subtree_files:
        for frac in dates_knonwn_fraction:
            for point in xrange(Npoints):

                if CLUSTER:
                    call = ['qsub', '-cwd', '-b','y',
                         '-l', 'h_rt=23:59:0',
                         #'-o', './stdout.txt',
                         #'-e', './stderr.txt',
                         '-l', 'h_vmem=50G',
                         './generate_flu_missingDates_dataset_run.py']
                else:
                    call = ['./generate_flu_missingDates_dataset_run.py']

                filename_suffix = "_Nk{}_{}".format(frac,point)

                arguments = [
                        out_dir,
                        subtree,
                        str(frac),
                        filename_suffix
                        ]
                call.extend(arguments)
                sp.call(call)


