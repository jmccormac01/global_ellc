"""
Take a corner plot pickle file and
make the plot
"""
import argparse as ap
import pickle
import corner

def arg_parse():
    p = ap.ArgumentParser()
    p.add_argument("pickle_file",
                   help="name of file to load",
                   type=str)
    p.add_argument("nsteps",
                   help="number of steps in MCMC run",
                   type=int)
    p.add_argument("nwalkers",
                   help="number of walkers in MCMC run",
                   type=int)
    return p.parse_args()

if __name__ == "__main__":
    # get config info
    args = arg_parse()

    # load the file
    with open(args.pickle_file, 'rb') as pf:
        res = pickle.load(pf)

    initials, labels, samples = res

    # plot the data
    fig = corner.corner(samples,
                        labels=labels,
                        truths=initials,
                        plot_contours=False)
    fig.savefig(f'corner_{args.nsteps}steps_{args.nwalkers}walkers.png')
    fig.clf()
