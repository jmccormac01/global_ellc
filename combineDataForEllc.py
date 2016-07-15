"""
Code to combine multiple lcs into one ellc input file

To do:
    Confirm with PM the flag param, use 1 for now
"""
import sys
import glob as g
import argparse as ap

def argParse():
    description="""
               Code for combining lcs into an ellc input file\n
               Run this in the folder with the lcs to combine
                """
    parser = ap.ArgumentParser(description=description)
    parser.add_argument('file_ids', help='wildcard name of files to combine')
    parser.add_argument('output_name', help='name of combined output file')
    parser.add_argument('--flag', help='flag to use for data points', default=1, type=int)
    return parser.parse_args()

# columns of BJD, Mag, Mag_err in NITES output
nites_cols = [2, 5, 6]

if __name__ = '__main__':
    args = argParse()
    t = g.glob(args.file_ids)
    if len(t) > 0:
        out_t, out_m, out_e, out_flag = [], [], [], []
        f = open(args.output_name, 'w')
        for filename in t:
            t, m, e = np.loadtxt(filename, usecols=nites_cols, unpack=True)
            for j, k, l in zip(t, m, e):
                f.write("{0:.8f}\t{1:.5f}\t{2:.5f}\t{3:d}\n".format(j, k, l, args.flag))
        f.close()
    else:
        print('There are no lcs to combine with ID {0:s}'.format(args.file_ids))
        sys.exit(1)

