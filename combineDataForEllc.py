"""
Code to combine multiple lcs into one ellc input file
"""
import sys
import glob as g
import argparse as ap
import numpy as np

def argParse():
    description="""
               Code for combining lcs into an ellc input file.
               Run this in the folder with the lcs to combine
                """
    parser = ap.ArgumentParser(description=description)
    parser.add_argument('file_ids', help='wildcard name of files to combine')
    parser.add_argument('output_name', help='name of combined output file')
    return parser.parse_args()

# columns of BJD, Mag, Mag_err, flag in NITES output
# flags are created from addFlagsDummyPoints.py
nites_cols = [2, 5, 6, 7]

if __name__ == '__main__':
    args = argParse()
    t = g.glob(args.file_ids)
    if len(t) > 0:
        outfile = open(args.output_name, 'w')
        for filename in t:
            t, m, e, f = np.loadtxt(filename, usecols=nites_cols, unpack=True)
            for i, j, k, l  in zip(t, m, e, f):
                outfile.write("{0:.8f}\t{1:.5f}\t{2:.5f}\t{3:d}\n".format(i, j, k, int(l)))
        outfile.close()
    else:
        print('There are no lcs to combine with ID {0:s}'.format(args.file_ids))
        sys.exit(1)

