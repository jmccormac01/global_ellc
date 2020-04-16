"""
Take an output csv file containing the MCMC chain from ellc and plot it

Plot the following:
    1. The walkers
    2. The corner plot (see if we can tweak the hist bins)
    3 The best fitting ellc model?
"""
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse as ap
import corner

def arg_parse():
    p = ap.ArgumentParser()
    p.add_argument('chain',
                   help='filename of MCMC chain file')
    return p.parse_args()

if __name__ == "__main__":
    args = arg_parse()
    # grab the samples quickly with numpy
    samples = np.loadtxt(args.chain, delimiter=',')
    # grab the top line to get the col_ids
    col_ids = open(args.chain).readline().replace('#', '').split()
