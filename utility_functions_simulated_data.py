#!/usr/bin/env python
"""
This module defines some  utility functions,
which are needed to operate the simulated data workflow.
"""
import treetime
from Bio import Phylo, AlignIO, Align
import os, sys
import treetime
import numpy as np
from external_binaries import *
from utility_functions_general import internal_regress, remove_polytomies, parse_lsd_output
import subprocess

NEAREST_DATE = 2016.5

def evolve_seq(treefile, basename, mu=0.0001, L=1000, mygtr = treetime.GTR.standard('jc')):
    """
    Generate a random sequence of a given length, and evolve it on the tree

    Args:
     - treefile: filename for the tree, on which a sequence should be evolved.
     - basename: filename prefix to save alignments.
     - mu: mutation rate. The units of the mutation rate should be consistent with
     the tree branch length
     - L: sequence length.
     - mygtr: GTR model for sequence evolution

    """
    from treetime import seq_utils
    from Bio import Phylo, AlignIO
    import numpy as np
    from itertools import izip

    mygtr.mu = mu
    tree = Phylo.read(treefile, 'newick')
    tree.root.ref_seq = np.random.choice(mygtr.alphabet, p=mygtr.Pi, size=L)
    print ("Started sequence evolution...")
    mu_real = 0.0
    n_branches = 0
    #print ("Root sequence: " + ''.join(tree.root.ref_seq))
    for node in tree.find_clades():
        for c in node.clades:
            c.up = node
        if hasattr(node, 'ref_seq'):
            continue
        t = node.branch_length
        p = mygtr.propagate_profile( seq_utils.seq2prof(node.up.ref_seq, mygtr.profile_map), t)
        # normalie profile
        p=(p.T/p.sum(axis=1)).T
        # sample mutations randomly
        ref_seq_idxs = np.array([int(np.random.choice(np.arange(p.shape[1]), p=p[k])) for k in np.arange(p.shape[0])])
        node.ref_seq = np.array([mygtr.alphabet[k] for k in ref_seq_idxs])
        node.ref_mutations = [(anc, pos, der) for pos, (anc, der) in
                            enumerate(izip(node.up.ref_seq, node.ref_seq)) if anc!=der]
        #print (node.name, len(node.ref_mutations))
        mu_real += 1.0 * (node.ref_seq != node.up.ref_seq).sum() / L
        n_branches += t
    mu_real /= n_branches
    print ("Mutation rate is {}".format(mu_real))
    records = [Align.SeqRecord(Align.Seq("".join(k.ref_seq)), id=k.name, name=k.name)
        for k in tree.get_terminals()]
    full_records = [Align.SeqRecord(Align.Seq("".join(k.ref_seq)), id=k.name, name=k.name)
        for k in tree.get_terminals()]
    #import ipdb; ipdb.set_trace()
    aln = Align.MultipleSeqAlignment(records)
    full_aln = Align.MultipleSeqAlignment(full_records)
    print ("Sequence evolution done...")

    # save results
    AlignIO.write(aln, basename+'.aln.ev.fasta', 'fasta')
    AlignIO.write(full_aln, basename+'.aln.ev_full.fasta', 'fasta')

    return aln, full_aln, mu_real

def _create_random_gtr(mu, alphabet='nuc'):
    """
    Create random GTR model
    """
    alph = treetime.seq_utils.alphabets[alphabet]
    pis = np.random.rand(alph.shape[0])
    pis /= np.sum(pis)

    W = np.random.rand(alph.shape[0], alph.shape[0])
    W = W/W.sum()
    return treetime.GTR.custom(mu, pis, W, alphabet=alphabet)

def _evolve_sequence(tree, L, gtr):
    """
    Produce random sequence of a given length L, evolve it on a given tree
    using the given gtr model.
    """
    if isinstance(tree, str):
        tree = Phylo.read(tree, 'newick')

    root_seq = np.random.choice(gtr.alphabet, p=gtr.Pi, size=1000)
    tree.root.ref_seq = root_seq
    print ("Started sequence evolution...")

    for node in tree.find_clades():

        for c in node.clades:
            c.up = node

        if hasattr(node, 'ref_seq'):
            continue

        t = node.branch_length
        p = gtr.propagate_profile(treetime.seq_utils.seq2prof(node.up.ref_seq, gtr.profile_map), t)
        # normalie profile
        p=(p.T/p.sum(axis=1)).T

        # sample mutations randomly
        ref_seq_idxs = np.array([int(np.random.choice(np.arange(p.shape[1]), p=p[k])) for k in np.arange(p.shape[0])])
        node.ref_seq = np.array([gtr.alphabet[k] for k in ref_seq_idxs])

    records = [Align.SeqRecord(Align.Seq("".join(k.ref_seq)), id=k.name, name=k.name)
        for k in tree.get_terminals()]

    aln = Align.MultipleSeqAlignment(records)
    #full_aln = Align.MultipleSeqAlignment(full_records)
    print ("Sequence evolution done...")
    return root_seq, aln

def gtr_comparison(basename, mu_avg_t, L=1e3):
    """
    Compare the two GTR models.
    """

    def _get_avg_branch_len(treefile):

        n_b, t_b = 0, 0
        tt = Phylo.read(treefile, 'newick')
        for clade in tt.find_clades():
            n_b += 1
            t_b += clade.branch_length
        return t_b / n_b



    original_tree = basename + ".nwk"
    avg_t = _get_avg_branch_len(original_tree)

    # mutation rate from the mu*t product
    mu = mu_avg_t / avg_t

    original_gtr = _create_random_gtr(mu, alphabet='nuc')
    root_seq, aln = _evolve_sequence(original_tree, L=L, gtr=original_gtr)

    myTree = treetime.TreeAnc(original_tree, aln, treetime.GTR.standard(model='JC69'))
    myTree.optimize_seq_and_branch_len(reuse_branch_len=False, infer_gtr=False)
    myTree.optimize_seq_and_branch_len(infer_gtr=True)

    KL = (original_gtr.Pi * np.log(original_gtr.Pi / myTree.gtr.Pi)).sum()

    # FIXME diagonal
    np.sum((original_gtr.W - myTree.gtr.W)**2)
    print(original_gtr.Pi, myTree.gtr.Pi)
    print(original_gtr.W, myTree.gtr.W)

    return KL

def run_treetime(basename, outfile, fasttree=False, failed=None, **kwargs):
    """
    Infer the dates of the internal nodes using the TreeTime package.
    Append results to the given file.

    Args:

     - basename(str): file prefix, which is resolved in the alignment path (by
     adding '.nuc.fasta') and to the tree filename (by adding '.opt.nwk' or similar).

     - outfile(str): output file to save results. The results will be written in
     append mode.

     - fasttree(bool): whether to use fasttree-generated tree (<basename>.ft.nwk)
     or not (use <basename>.opt.nwk)

     - failed(list or None): in not None, in case of treetime failure, the basename
     will be appended to the list for further analysis

    **Kwargs:

     - all arguments will be passe down to the treetime object run function.
    """
    if fasttree:
        treefile = basename + ".ft.nwk"
        outtree = basename + ".treetime.ft.nwk"
    else:
        treefile = basename + ".opt.nwk"
        outtree = basename + ".treetime.nwk"
    aln = basename+'.nuc.fasta'
    Tmrca, dates = dates_from_ffpopsim_tree(Phylo.read(treefile, "newick"))
    myTree = treetime.TreeTime(gtr='Jukes-Cantor', tree = treefile,
        aln = aln, verbose = 4, dates = dates, debug=False)

    print ("Use input branch length is set to: {}".format(kwargs['use_input_branch_length']))
    myTree.run(root='best', **kwargs)
    Phylo.write(myTree.tree, outtree, 'newick')

    if not os.path.exists(outfile):
        try:
            with open(outfile, 'w') as of:
                of.write("#File,Tmrca_real,Tmrca,Mu,R^2(initial_clock),R^2(internal_nodes)\n")
        except:
            pass

    with open(outfile, 'a') as of:
        of.write("{},{},{},{},{},{}\n".format(
            basename,
            str(Tmrca),
            str(myTree.tree.root.numdate),
            str(myTree.date2dist.clock_rate),
            str(myTree.date2dist.r_val),
            str(internal_regress(myTree)) ))

    return myTree

def _create_date_file_from_ffpopsim_tree(treefile, datesfile):
    """
    Read dates of the terminal nodes in the simulated tree, and save them into the
    dates file in LSD format
    """
    Tmrca, dates = dates_from_ffpopsim_tree(treefile)
    with open(datesfile, 'w') as df:
        df.write(str(len(dates)) + "\n")
        df.write("\n".join([str(k) + "\t" + str(dates[k]) for k in dates]))
    return Tmrca, dates

def run_lsd(treefile, datesfile, outfile, res_file):
    """
    Infer the dates of the internal nodes using the LSD package.
    Append results to the given file.
    Args:

     - treefile: the name for the input tree in newick format.

     - datesfile:  Filename, where to store the dates in the LSD format. If the
     file does not exist, it will be created automatically.

     - outfile: prefix for the filename, where LSD will store the results (different
     trees, output logs, etc.).

     - res_file: file where to write the formatted result string. This file is
     then parsed by the processing scripts to produce comparison with TreeTime
    """
    outfile = outfile.replace(".nwk", ".res.txt")
    Tmrca, dates = _create_date_file_from_ffpopsim_tree(treefile, datesfile)
    # call LSD binary
    call = [LSD_BIN, '-i', treefile, '-d', datesfile, '-o', outfile,
            '-r', 'a', '-c', 'v']

    subprocess.call(call)
    print ("LSD Done!")

    tmrca, mu, objective = parse_lsd_output(outfile)
    if float(mu) <= 0:
        return

    if not os.path.exists(res_file):
        try:
            with open(res_file, 'w') as of:
                of.write("#File,Tmrca_real,Tmrca,Mu,objective\n")
        except:
            pass

    with open(res_file, "a") as of:
        of.write("{},{},{},{},{}\n".format(treefile, str(Tmrca), tmrca, mu, objective))

def run_ffpopsim_simulation(L, N, SAMPLE_VOL, SAMPLE_NUM, SAMPLE_FREQ, MU, res_dir, res_suffix, failed=None, **kwargs):
    """
    Run simulation with FFPopSim package and perform the data preprocessing.
    The FFpopSim produces phylogenetic tree and alignment in the binary (0-1) form.
    The tree branch lengths are in units of time, expressed in generations.
    Returns:
     - basename: base name of the files, where the results are stored. The file
     suffixes are added for each file type separately.
    """

    sys.stdout.write("Importing modules...")

    # check the output location
    if not os.path.exists(res_dir) or not os.path.isdir(res_dir):
        os.makedirs(res_dir)

    # run ffpopsim
    sys.stdout.write("Running FFpopSim...")
    basename = _run_ffpopsim(L=L, N=N,
                    SAMPLE_NUM=SAMPLE_NUM,
                    SAMPLE_FREQ=SAMPLE_FREQ,
                    SAMPLE_VOL=SAMPLE_VOL,
                    MU=MU,
                    res_dir=res_dir, res_suffix=res_suffix)

    # pot-process the results
    if 'optimize_branch_len' in kwargs:
        optimize_branch_len = kwargs['optimize_branch_len']
    else:
        optimize_branch_len = True
    _ffpopsim_tree_aln_postprocess(basename, optimize_branch_len=optimize_branch_len)
    print ("Done clusterSingleFunc")
    return basename

def _run_ffpopsim(L=100, N=100, SAMPLE_NUM=10, SAMPLE_FREQ=5, SAMPLE_VOL=15, MU=5e-5, res_dir="./", res_suffix=""):
    """
    Simple wrapper function to call FFpopSim binary in a separate subprocess
    """

    basename = "FFpopSim_L{}_N{}_Ns{}_Ts{}_Nv{}_Mu{}".format(str(L), str(N),
                str(SAMPLE_NUM), str(SAMPLE_FREQ), str(SAMPLE_VOL), str(MU))

    basename = os.path.join(res_dir, basename)
    if res_suffix != "":
        basename = basename + "_" + res_suffix


    call = [FFPOPSIM_BIN, L, N, SAMPLE_NUM, SAMPLE_FREQ, SAMPLE_VOL, MU, basename]
    os.system(' '.join([str(k) for k in call]))

    return basename

def _ffpopsim_tree_aln_postprocess(basename, optimize_branch_len=False, prefix='Node/'):

    """
    Given the raw data produced ini the FFPopSim simulation, perform the preliminary
    data processing. This includes converting the alignment in nucleotide notation, where
    0 is converted to 'A', 1 is converted to 'C'. The tree branch lengths are been
    optimized. The tree is checked to have no multiple mergers (the multiple mergers are
    resolved randomly). The nodes of tree are named in a unified way and if there are
    non-unique names found, they resolved by adding additional suffixes
    """

    def ffpopsim_aln_to_nuc(basename):

        with open(basename + ".bin.fasta", 'r') as inf:
            ss = inf.readlines()

        with open(basename + ".nuc.fasta", 'w') as of:
            conversion = None
            for s in ss:
                if s.startswith(">"):
                    of.write(s)
                else:
                    s_array = np.fromstring(s.strip(), 'S1')
                    if conversion is None:
                        conversion = np.random.randint(2, size=s_array.shape)
                    if (len(s_array)):
                        nuc_seq = np.zeros_like(s_array, dtype='S1')
                        nuc_seq[:]='A'
                        replace_ind = np.array(((s_array=='1')&(conversion))|((s_array=='0')&(~conversion)), 'bool')
                        nuc_seq[replace_ind] = 'C'
                        of.write("".join(nuc_seq)+'\n')

    def generation_from_node_name(name):
        try:
            return int(name.split("_")[1])
        except:
            return -1


    from collections import Counter

    t = Phylo.read(basename + ".nwk", "newick")
    if len(t.root.clades)==1:
        t.root = t.root.clades[0]
    t = remove_polytomies(t)
    cs = [k for k in t.find_clades() if len(k.clades)==1]
    for clade in cs:
        t.collapse(clade)
    t.root.branch_length = 1e-5
    t.ladderize()
    # remove duplicate names:
    t_counter = Counter(t.find_clades())
    for clade in t.find_clades():
        if clade.name is None:
            continue
        if t_counter[clade] > 1 :
            name_suffix = t_counter[clade]
            clade.name += "/"+str(name_suffix)
            t_counter[clade] -= 1

        if clade.name is not None and not clade.name.startswith(prefix):
            clade.name = prefix + clade.name

    max_generation = np.max([generation_from_node_name(k.name) for k in t.find_clades()])
    min_generation = generation_from_node_name(t.root.name)
    assert(max_generation > min_generation and min_generation > 0)

    for clade in t.find_clades():
        node_gen = generation_from_node_name(clade.name)
        if clade.name is not None and not '_DATE_' in clade.name and node_gen != -1:
            #clade.name += "_DATE_%d"%node_gen # + str(2016.5 - (max_generation - node_gen))
            clade.name += "_DATE_" + str(NEAREST_DATE - (max_generation - node_gen))


    Phylo.write(t, basename + ".nwk", "newick")

    # prepare alignment
    ffpopsim_aln_to_nuc(basename)
    aln = AlignIO.read(basename + ".nuc.fasta", "fasta")
    names = [k.id for k in aln]
    aln_counter = Counter(names)
    for k in aln:
        key = k.id
        # remove duplicate names
        if (aln_counter[key] > 1):
            k.id = k.id + "/" + str(aln_counter[k.id])
            k.name = k.id
            aln_counter[key] -= 1
        # add name prefix
        if not k.id.startswith(prefix):
            k.id = prefix + k.id
            k.name = k.id

        node_gen = generation_from_node_name(k.id)
        if not "_DATE_" in k.id and node_gen != -1:
            k.id += "_DATE_" + str( NEAREST_DATE- (max_generation - node_gen))
            k.name = k.id

    AlignIO.write(aln, basename + ".nuc.fasta", "fasta")

    if optimize_branch_len:
        import treetime
        gtr = treetime.GTR.standard('jc')
        tanc = treetime.TreeAnc(aln=aln,tree=t,gtr=gtr)
        tanc.optimize_seq_and_branch_len(reuse_branch_len=False,prune_short=False,infer_gtr=False)
        Phylo.write(tanc.tree, basename+".opt.nwk", "newick")

def reconstruct_fasttree(basename, optimize_branch_len=False):
    """
    Run the fast tree reconstruction given the alignment produced by FFPopSim
    simulations
    """
    def fasttree_post_process(aln, basename, optimize_branch_len):
        ffpopsim_treefile = basename + ".nwk"
        treefile = basename + ".ft.nwk"
        tree = Phylo.read(treefile, 'newick')
        tree = remove_polytomies(tree)
        tree.ladderize()
        Tmrca,dates = dates_from_ffpopsim_tree(ffpopsim_treefile)
        tree.root.name = "FFPOPsim_Tmrca_DATE_" + str(Tmrca)

        # optimize branch lengths (mainly, to remove the fasttree zero-lengths artefacts)
        if optimize_branch_len:
            gtr = treetime.GTR.standard('jc')
            tanc = treetime.TreeAnc(tree=tree, aln=aln, gtr=gtr)
            tanc.optimize_seq_and_branch_len(reuse_branch_len=True,prune_short=False,infer_gtr=False,max_iter=5)
            return tanc.tree
        else:
            return tree

    fasta = basename + ".nuc.fasta"
    outfile = fasta.replace('.nuc.fasta', ".ft.nwk")
    # call FastTree to reconstruct newick tree
    call = [FAST_TREE_BIN, "-nt", fasta, ">", outfile]
    os.system(' '.join([str(k) for k in call]))
    tree = fasttree_post_process(fasta, basename, optimize_branch_len=optimize_branch_len)
    os.remove(outfile)
    Phylo.write(tree, outfile, 'newick')

def generations_from_ffpopsim_tree(t):
    """
    """
    if isinstance(t, str):
        t = Phylo.read(t, 'newick')
    try:
        generations = {}
        for clade in t.get_terminals():
            if "_DATE_"not in clade.name:
                continue
            generations[clade.name] = float(clade.name.split("_")[1])
        Gen_mrca = float (t.root.name.split("_")[1])
        return Gen_mrca, generations
    except:
        return 0, {}

def dates_from_ffpopsim_tree(t):
    """
    Args:
     - t: tree filename or BioPyhton tree object
    """
    if isinstance(t, str):
        t = Phylo.read(t, 'newick')
    try:
        dates = {}
        for clade in t.get_terminals():
            if "_DATE_"not in clade.name:
                continue
            dates[clade.name] = float(clade.name.split("_")[-1])
        Tmrca = float (t.root.name.split("_")[-1])
        return Tmrca, dates
    except:
        return NEAREST_DATE, {}

def run_beast(basename, out_dir, res_file, fast_tree=True):
    """
    From basename, compose names for the tree, dates and lignment, and call
    run_beast from the beast_utilities.py module
    """

    try: # if running in parallel, migh be simultaneous creation of the same dir from different threads
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
    except:
        pass
    import utility_functions_beast as beast_utils

    if fast_tree:
        treename = basename + ".ft.nwk"
    else:
        treename = basename + ".opt.nwk"

    alnname = basename + ".nuc.fasta"
    Tmrca, dates = dates_from_ffpopsim_tree(Phylo.read(treename, "newick"))
    beast_res_prefix = os.path.join(out_dir, os.path.split(basename)[-1])

    # define log post-processing:
    def process_results(log_file):
        """
        Read BEAST log file, extract results and save them to the resulting csv file
        """
        df = beast_utils.read_beast_log(log_file, np.max(dates.values()))
        if df is None or df.shape[0] < 200 :
            print ("Beast log {} is corrupted or BEAST run did not finish".format(log_file))
            return

        Sim_Mu = float(log_file.split("/")[-1].split('_')[6][2:])
        Ns = int(log_file.split("/")[-1].split('_')[3][2:])
        Ts = int(log_file.split("/")[-1].split('_')[4][2:])
        N = int(log_file.split("/")[-1].split('_')[2][1:])
        T = Ns * Ts
        Nmu = N * Sim_Mu

        inferred_LH = df['likelihood'][-50:].mean()
        inferred_LH_std = df['likelihood'][-50:].std()
        inferred_Tmrca = df['treeModel.rootHeight'][-50:].mean()
        inferred_Tmrca_std = df['treeModel.rootHeight'][-50:].std()
        inferred_Mu = df['clock.rate'][-50:].mean()
        inferred_Mu_std = df['clock.rate'][-50:].std()

        #dTmrca = -(Sim_Tmrca[-1] - Tmrca[-1])
        #dMu =  Sim_Mu[-1] - Mu[-1]

        res_str = "{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
            os.path.split(basename)[-1],
            N,Tmrca,Sim_Mu,Ns,Ts,T,Nmu,
            inferred_LH,inferred_LH_std,inferred_Tmrca,inferred_Tmrca_std,inferred_Mu,inferred_Mu_std)

        if not os.path.exists(res_file):
            try:
                with open(res_file, 'w') as of:
                    of.write("#Filename,PopSize,Tmrca_real,ClockRate_real,SamplesNum,SampleFreq,TotEvoTime(Ns*Ts),Nmu,LH,LH_std,Tmrca,Tmrca_std,Mu,Mu_std\n")
            except:
                pass

        with open(res_file, 'a') as outfile:
            outfile.write(res_str)

    beast_utils.run_beast(treename, alnname, dates, beast_res_prefix,
        template_file="./resources/beast/template_bedford_et_al_2015.xml",
        log_post_process=process_results)

if __name__ == "__main__":
    pass
