import pandas
import numpy as np
#from scipy.stats import linregress
import matplotlib.pyplot as plt
import os
#import shutil
#from Bio import Phylo

import utility_functions_flu as flu_utils
import utility_functions_beast as beast_utils

from plot_defaults import *

##  Read datasets as-is
def read_lsd_dataset(fname):
    """
    TODO
    """
    lsd_cols = ['File', 'N', 'Tmrca_sim', 'mu_sim', 'Runtime', 'objective']
    lsd_df = pandas.read_csv(fname, names=lsd_cols, header=0)
    return lsd_df

def read_treetime_dataset(fname):
    """
    TODO
    """
    cols = ['File', 'N', "Tmrca_sim", "mu_sim", "R2_leaves", "R2_internal", "Runtime"]
    df = pandas.read_csv(fname, names=cols,header=0)
    return df

def read_beast_dataset(fname):
    """
    TODO
    """
    cols = ['File', 'N', 'LH', 'LH_std', 'Tmrca', 'Tmrca_std', 'Mu', 'Mu_std']
    df = pandas.read_csv(fname, names=cols,header=0)
    return df

def IQD(a):
    from scipy.stats import scoreatpercentile
    return scoreatpercentile(a,75) - scoreatpercentile(a,25)

def make_beast_pivot(df):

    Tmrca_median = []
    Tmrca_err = []
    LH_median = []
    LH_err = []
    Mu_median = []
    Mu_err = []

    Ns = df["N"].unique()

    Nsidx = np.ones(Ns.shape, dtype=bool)
    for idx, N in enumerate(Ns):
        Nidx = df["N"] == N
        if Nidx.sum() == 0:
            Nsidx[idx] = False
            continue
        Tmrca_median.append(df[Nidx]["Tmrca"].median())
        Tmrca_err.append(IQD(df[Nidx]["Tmrca"]))
        Mu_median.append(df[Nidx]["Mu"].median())
        Mu_err.append(IQD(df[Nidx]["Mu"]))
        LH_median.append(df[Nidx]["LH"].median())
        LH_err.append(IQD(df[Nidx]["LH"]))

    res = pandas.DataFrame({
        "Ns" : Ns[Nsidx],
        "Tmrca_median" : Tmrca_median,
        "Tmrca_err" : Tmrca_err,
        "Mu_median" : Mu_median,
        "Mu_err" : Mu_err,
        "LH_median" : LH_median,
        "LH_err" : LH_err
        })

    return res

def make_treetime_pivot(df):

    Tmrca_median = []
    Tmrca_err = []
    Mu_median = []
    Mu_err = []
    Runtime_median = []
    Runtime_err = []

    Ns = df["N"].unique()

    Nsidx = np.ones(Ns.shape, dtype=bool)
    for idx, N in enumerate(Ns):
        Nidx = df["N"] == N
        if Nidx.sum() == 0:
            Nsidx[idx] = False
            continue
        Tmrca_median.append(df[Nidx]["Tmrca_sim"].median())
        Tmrca_err.append(IQD(df[Nidx]["Tmrca_sim"]))
        Mu_median.append(df[Nidx]["mu_sim"].median())
        Mu_err.append(IQD(df[Nidx]["mu_sim"]))
        Runtime_median.append(df[Nidx]["Runtime"].median())
        Runtime_err .append(IQD(df[Nidx]["Runtime"]))

    res = pandas.DataFrame({
        "Ns" : Ns[Nsidx],
        "Tmrca_median" : Tmrca_median,
        "Tmrca_err" : Tmrca_err,
        "Mu_median" : Mu_median,
        "Mu_err" : Mu_err,
        "Runtime_median" : Runtime_median,
        "Runtime_err" : Runtime_err
        })
    return res

def make_lsd_pivot(df):

    Tmrca_median = []
    Tmrca_err = []
    Mu_median = []
    Mu_err = []
    Runtime_median = []
    Runtime_err = []

    Ns = df["N"].unique()
    Nsidx = np.ones(Ns.shape, dtype=bool)

    for idx, N in enumerate(Ns):
        Nidx = df["N"] == N
        if Nidx.sum() == 0:
            Nsidx[idx] = False
            continue

        Tmrca_median.append(df[Nidx]["Tmrca_sim"].median())
        Tmrca_err.append(IQD(df[Nidx]["Tmrca_sim"]))
        Mu_median.append(df[Nidx]["mu_sim"].median())
        Mu_err.append(IQD(df[Nidx]["mu_sim"]))
        Runtime_median.append(df[Nidx]["Runtime"].median())
        Runtime_err .append(IQD(df[Nidx]["Runtime"]))

    res = pandas.DataFrame({
        "Ns" : Ns[Nsidx],
        "Tmrca_median" : Tmrca_median,
        "Tmrca_err" : Tmrca_err,
        "Mu_median" : Mu_median,
        "Mu_err" : Mu_err,
        "Runtime_median" : Runtime_median,
        "Runtime_err" : Runtime_err
        })

    return res

## Plot statistics
def plot_res(what, tt=None, lsd=None, beast=None, save=True, suffix=None, scatter_points=True, **kwargs):

    if what == 'Tmrca':
        median = 'Tmrca_median'
        err = 'Tmrca_err'
        #title = "Estimated Tmrca as function of sample size\nLSD params: -{}".format(suffix)
        ylim = [2003,2011]

        ylabel = "T$\mathrm{_{mrca}}, [\mathrm{Year}]$"

    elif what == "Mu":
        median = 'Mu_median'
        err =  'Mu_err'
        ylim = [0,0.005]
        #title =  "Estimated substitution rate as function of sample size\nLSD params: -{}".format(suffix)

        ylabel = "substitution rate, [$\mathrm{Year}^{-1}$]"

    fig = plt.figure(figsize=onecolumn_figsize)
    axes = fig.add_subplot(111)
    axes.ticklabel_format(useOffset=False)
    axes.set_xscale('log')

    if tt is not None:
        if scatter_points:
            x, y = shift_point_by_markersize (axes, tt['Ns'], tt[median], markersize/2.0)
        else:
            x, y = tt['Ns'], tt[median]
        axes.errorbar(x, y, tt[err]/2, markersize=markersize, marker='o', c=tt_color, label='TreeTime')

    if lsd is not None:
        if scatter_points:
            x, y = shift_point_by_markersize (axes, lsd['Ns'], lsd[median], -1.*markersize/2.0)
        else:
            x, y = lsd['Ns'], lsd[median]
        axes.errorbar(x, y, lsd[err]/2, markersize=markersize, marker='o', c=lsd_color, label='LSD')

    if beast is not None:
        #  beast points stay in the center
        x, y = beast['Ns'], beast[median]
        # if scatter_points:
        #     shift_point_by_markersize (axes, beast['Ns'], beast[median], -1.*markersize/2.0)
        # else:
        #     x, y = beast['Ns'], beast[median]
        axes.errorbar(x, y, beast[err]/2, markersize=markersize, marker='o', c=beast_color, label='BEAST')

    axes.grid('on')
    axes.legend(loc=0,fontsize=legend_fs)
    axes.set_ylabel(ylabel, fontsize=label_fs)
    axes.set_xlabel("number of sequences", fontsize=label_fs)
    axes.set_ylim(ylim)
    #axes.set_title(title)

    for label in axes.get_xticklabels():
            label.set_fontsize(tick_fs)
    for label in axes.get_yticklabels():
            label.set_fontsize(tick_fs)

    if save:
        fig.savefig("./figs/fluH3N2_subtrees_{}.svg".format(what))
        fig.savefig("./figs/fluH3N2_subtrees_{}.png".format(what))
        fig.savefig("./figs/fluH3N2_subtrees_{}.pdf".format(what))

if __name__ == "__main__":

    PLOT_TREETIME = True
    PLOT_LSD = True
    PLOT_BEAST = True
    SAVE_FIG=True

    ##
    ##  Specify location of the CSV tables with results
    ##
    res_dir = './flu_H3N2/subtree_samples'
    treetime_res_file = os.path.join(res_dir, 'treetime_res.csv')
    lsd_res_file = os.path.join(res_dir, 'lsd_res.csv')
    beast_res_file = os.path.join(res_dir, 'beast_res.csv')


    ##
    ## Read datasets and make pivot tablespivots
    ##
    if PLOT_TREETIME:
        tt_df = make_treetime_pivot(read_treetime_dataset(treetime_res_file))
        tt_df = tt_df.sort(columns='Ns')
    else:
        tt_df = None

    if PLOT_LSD:
        lsd_df = make_lsd_pivot(read_lsd_dataset(lsd_res_file))
        lsd_df = lsd_df.sort(columns='Ns')
    else:
        lsd_df = None

    if PLOT_BEAST:
        beast =   make_beast_pivot(read_beast_dataset(beast_res_file))
    else:
        beast=None

    ##
    ## Plot the results:
    ##
    plot_res('Tmrca', tt=tt_df, lsd=lsd_df, beast=beast, save=SAVE_FIG)
    plot_res('Mu', tt=tt_df, lsd=lsd_df, beast=beast, save=SAVE_FIG)

