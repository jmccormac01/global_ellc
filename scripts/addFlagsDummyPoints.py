import argparse as ap
import numpy as np
import matplotlib.pyplot as plt

# pylint: disable = invalid-name

def argParse():
    """
    Parse the command line args
    """
    parser = ap.ArgumentParser()
    parser.add_argument('filename',
                        help='file to edit')
    parser.add_argument('--dummy_enddate',
                        help='JD of last dummy data point',
                        type=float)
    parser.add_argument('--flag',
                        help='flag to use for real data points',
                        default=1,
                        type=int)
    return parser.parse_args()

if __name__ == '__main__':
    args = argParse()
    t1, t2, t3, f, fe, m, me = np.loadtxt(args.filename,
                                          usecols=[0, 1, 2, 3, 4, 5, 6],
                                          unpack=True)
    # add the normal flags
    flags = np.ones(len(t1))* args.flag
    header = "JD-MID  HJD-MID  BJD_TDB-MID  FLUX  FLUX_ERR  MAG  MAG_ERR  ELLC_FLAG"
    fmt = "%.8f  %.8f  %.8f  %.4f  %.4f  %.4f  %.4f  %d"
    np.savetxt(args.filename,
               np.c_[t1, t2, t3, f, fe, m, me, flags],
               fmt=fmt,
               header=header)

    # add the dummy points
    if args.dummy_enddate:
        t1, t2, t3, f, fe, m, me, flag = np.loadtxt(args.filename, unpack=True)
        d = np.average(np.diff(t1))
        n_dummy_points = int((args.dummy_enddate - t1[-1]) / d)
        if n_dummy_points <= 0:
            print('Enter an end date after the last data point!')
            sys.exit(1)
        else:
            print('Adding {0:d} dummy points with flag=-1'.format(n_dummy_points))
            outfile = open(args.filename, 'a')
            for i in range(1, n_dummy_points+1):
                outfile.write('{0:.8f}  {1:.8f}  {2:.8f}  1.0000  0.0001  0.0000  0.0001  -1\n'.format(t1[-1]+(i*d), t2[-1]+(i*d), t3[-1]+(i*d)))

    
