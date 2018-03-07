"""
Take in a swasp light curve and convert
the values to diff flux. Also add on 2450000
"""
import sys
import argparse as ap
import numpy as np
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt

def argParse():
    """
    Parse the command line arguments
    """
    p = ap.ArgumentParser()
    p.add_argument('infile',
                   help='name of the input file')
    p.add_argument('--plot',
                   help='plot the conversion?',
                   action='store_true')
    p.add_argument('--phase',
                   help='phase the data?',
                   action='store_true')
    return p.parse_args()

if __name__ == "__main__":
    args = argParse()
    t, m, e = np.loadtxt(args.infile, usecols=[0, 1, 2], unpack=True)
    t0 = int(t[0])
    if t0 < 2450000:
        t = t + 2450000
        t0 = int(t[0])
    f = 10**(m/-2.5)
    fe = (e*np.log(10)*f)/2.5
    if args.plot:
        fig, ax = plt.subplots(2, figsize=(10, 10),
                               sharex=True)
        ax[0].errorbar(t-t0, m, yerr=e, fmt='k.')
        ax[0].set_ylabel('Diff mags')
        ax[1].errorbar(t-t0, f, yerr=fe, fmt='k.')
        ax[1].set_ylabel('Flux ratio')
        ax[1].set_xlabel('HJD - {}'.format(t0))
        plt.show()
    if args.phase:
        a = raw_input('E and P set? (y | n): ')
        if a.lower() != 'y':
            sys.exit()
        E = 2455009.43174
        P = 0.8546747
        ph = ((t-E)/P)%1
        fig2, ax2 = plt.subplots(1, figsize=(10, 10))
        ax2.plot(ph, f, 'k.')
        ax2.plot(ph-1, f, 'k.')
        plt.show()
    outfile = "{}.flux".format(args.infile)
    np.savetxt(outfile, np.c_[t, f, fe], fmt='%.6f %.6f %.6f', header='hjd flux_ratio error')
