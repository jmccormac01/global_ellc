"""
Use this script along with previously calculated
system parameters to fit spotty data for the spot
params only. This is based off the emcee_combined.py
code and is stripped down to run on a per lc basis
to get the best fitting spot or spot/faculi params
"""
import sys
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
import numpy as np
import emcee
import corner
import ellc

# pylint: disable=invalid-name
# pylint: disable=unused-variable
# pylint: disable=no-member
# pylint: disable=redefined-outer-name
# pylint: disable=superfluous-parens

def light_curve_model(t_obs, t0, period, radius_1, radius_2,
                      sbratio, incl, f_s, f_c, a, q, ldc_1,
                      spots_1=None, spots_2=None):
    """
    Takes in the binary parameters and returns an ellc model
    for the light curve
    """
    lc_model = ellc.lc(t_obs=t_obs,
                       t_zero=t0,
                       period=period,
                       radius_1=radius_1,
                       radius_2=radius_2,
                       incl=incl,
                       sbratio=sbratio,
                       a=a,
                       q=q,
                       ldc_1=ldc_1,
                       ld_1='quad',
                       shape_1='sphere',
                       shape_2='sphere',
                       grid_1='default', # set these to default again later
                       grid_2='default',
                       f_c=f_c,
                       f_s=f_s,
                       spots_1=spots_1,
                       spots_2=spots_2)
    return lc_model

def lnprior(theta):
    """
    Add docstring
    """
    lc1_l1, lc1_b1, lc1_s1, lc1_f1, \
    lc1_l2, lc1_b2, lc1_s2, lc1_f2 = theta
    # need to force a spot and a facula
    if -90.0 <= lc1_l1 <= 90.0 and \
       -90.0 <= lc1_l2 <= 90.0 and \
       -90.0 <= lc1_b1 <= 90.0 and \
       -90.0 <= lc1_b2 <= 90.0 and \
       0.1 <= lc1_s1 <= 60 and \
       0.1 <= lc1_s2 <= 60 and \
       0.0 <= lc1_f1 <= 1.0 and \
       1.0 < lc1_f2 <= 2.0:
        return 0.0
    else:
        return -np.inf

def lnlike_sub(data_type, model, data, error):
    """
    Work out the log likelihood for a given subset of data
    """
    if data_type == 'phot':
        if True in np.isnan(model) or np.min(model) <= 0:
            lnlike = -np.inf
        else:
            inv_sigma2 = 1.0/(error**2)
            eq_p1 = (data-model)**2*inv_sigma2 - np.log(inv_sigma2)
            lnlike = -0.5*(np.sum(eq_p1) - np.log(len(data) + 1))
    elif data_type == 'rv':
        if True in np.isnan(model):
            lnlike = -np.inf
        else:
            inv_sigma2 = 1.0/(error**2)
            eq_p1 = (data-model)**2*inv_sigma2 - np.log(inv_sigma2)
            lnlike = -0.5*(np.sum(eq_p1) - np.log(len(data) + 1))
    else:
        print('UNKNOWN DATA_TYPE IN lnlike_sub, EXITING...')
        sys.exit(1)
    return lnlike

def lnlike(theta, x_lc1, y_lc1, yerr_lc1):
    """
    Work out the log likelihood for the proposed model
    """
    # unpack theta and pass parms to model
    lc1_l1, lc1_b1, lc1_s1, lc1_f1, \
    lc1_l2, lc1_b2, lc1_s2, lc1_f2 = theta

    # set the two ldcs into a list for ellc
    ldcs_1 = [in_ldc_1_1, in_ldc_1_2]

    # fixed parameters
    f_s = np.sqrt(in_ecc)*np.sin(in_omega*np.pi/180.)
    f_c = np.sqrt(in_ecc)*np.cos(in_omega*np.pi/180.)

    # light curve 1 likelihood function - spotty
    model_lc1 = light_curve_model(t_obs=x_lc1,
                                  t0=in_t0,
                                  period=in_period,
                                  radius_1=in_radius_1,
                                  radius_2=in_radius_2,
                                  sbratio=in_sbratio,
                                  a=in_a,
                                  q=in_q,
                                  incl=in_incl,
                                  f_s=f_s,
                                  f_c=f_c,
                                  ldc_1=ldcs_1,
                                  spots_1=[[lc1_l1], [lc1_b1], [lc1_s1], [lc1_f1]],
                                  spots_2=[[lc1_l2], [lc1_b2], [lc1_s2], [lc1_f2]])
    lnlike = lnlike_sub('phot', model_lc1, y_lc1, yerr_lc1)
    return lnlike

def lnprob(theta, x_lc1, y_lc1, yerr_lc1):
    """
    Add docstring
    """
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x_lc1, y_lc1, yerr_lc1)

if __name__ == "__main__":
    # fixed params from known models
    in_radius_1 = 0.029505    #solar radii
    in_radius_2 = 0.004606     #solar radii
    in_sbratio = 0.0           # fixed = set in lnlike
    in_q = 0.09615
    in_incl = 89.618
    in_t0 = 2453592.74546
    in_period = 16.95353
    in_ecc = 0.1603
    in_omega = 78.4529
    in_a = 31.2868           #solar radii
    in_ldc_1_1 = 0.4480
    in_ldc_1_2 = 0.1878

    # these are the guesses
    in_lc1_l1 = 31.0 # lc1_s1 spot params
    in_lc1_b1 = -5.0
    in_lc1_s1 = 5.0
    in_lc1_f1 = 0.8
    in_lc1_l2 = -20.0 # lc1_s2 spot params
    in_lc1_b2 = -5.0
    in_lc1_s2 = 10.0
    in_lc1_f2 = 1.2

    # list of initial guesses
    initial = [in_lc1_l1,
               in_lc1_b1,
               in_lc1_s1,
               in_lc1_f1,
               in_lc1_l2,
               in_lc1_b2,
               in_lc1_s2,
               in_lc1_f2]
    # used in plotting
    parameters = ['lc1_l1', 'lc1_b1', 'lc1_s1', 'lc1_f1',
                  'lc1_l2', 'lc1_b2', 'lc1_s2', 'lc1_f2']
    # set up the weights for the initialisation
    # these weights are used to scattter the walkers
    # if using a prior make sure they are not scattered
    # outside the range of the prior
    weights = [1e-2, 1e-2, 1e-2, 1e-2,
               1e-2, 1e-2, 1e-2, 1e-2]
    # check the lists are the same length
    assert len(initial) == len(parameters) == len(weights)

    # READ IN THE DATA
    datadir = '/Users/jmcc/Dropbox/EBLMs/J23431841'
    outdir = '{}/output'.format(datadir)
    # other night needing fit later
    #lc1_file = 'NITES_J234318.41_20130923_Clear_F2.lc.txt'
    lc1_file = 'NITES_J234318.41_20120829_Clear_F1.lc.txt'
    x_lc1, y_lc1, yerr_lc1 = np.loadtxt('{}/{}'.format(datadir, lc1_file),
                                        usecols=[2, 3, 4], unpack=True)

    # set up the sampler
    ndim = len(initial)
    nwalkers = 4*len(initial)
    nsteps = 500
    # set up the starting positions
    pos = [initial + weights*np.random.randn(ndim) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(x_lc1, y_lc1, yerr_lc1))

    # run the production chain
    print("Running MCMC...")
    sampler.run_mcmc(pos, nsteps, rstate0=np.random.get_state())
    print("Saving chain...")
    np.savetxt('{}/chain_{}steps_{}walkers.csv'.format(outdir, nsteps, nwalkers),
               np.c_[sampler.chain.reshape((-1, ndim))],
               delimiter=',', header=','.join(parameters))
    print("Done.")

    # plot and save the times series of each parameter
    for i, (initial_param, label) in enumerate(zip(initial, parameters)):
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.plot(sampler.chain[:, :, i].T, color="k", alpha=0.4)
        ax.axhline(initial_param, color="#888888", lw=2)
        ax.set_ylabel(label)
        ax.set_xlabel('step number')
        fig.savefig('{}/chain_{}steps_{}walkers_{}.png'.format(outdir,
                                                               nsteps,
                                                               nwalkers,
                                                               label))

    # calculate the most likely set of parameters ###
    # get user to input the burni. period, after they
    # have seen the time series of each parameter
    burnin = int(raw_input('Enter burnin period: '))
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

    # most likely set of parameters
    lc1_l1 = np.median(samples[:, 0])
    lc1_b1 = np.median(samples[:, 1])
    lc1_s1 = np.median(samples[:, 2])
    lc1_f1 = np.median(samples[:, 3])
    lc1_l2 = np.median(samples[:, 4])
    lc1_b2 = np.median(samples[:, 5])
    lc1_s2 = np.median(samples[:, 6])
    lc1_f2 = np.median(samples[:, 7])

    print(u'lc1_l1 = {} \u00B1 {}'.format(lc1_l1, np.std(samples[:, 0])))
    print(u'lc1_b1 = {} \u00B1 {}'.format(lc1_b1, np.std(samples[:, 1])))
    print(u'lc1_s1 = {} \u00B1 {}'.format(lc1_s1, np.std(samples[:, 2])))
    print(u'lc1_f1 = {} \u00B1 {}'.format(lc1_f1, np.std(samples[:, 3])))
    print(u'lc1_l2 = {} \u00B1 {}'.format(lc1_l2, np.std(samples[:, 4])))
    print(u'lc1_b2 = {} \u00B1 {}'.format(lc1_b2, np.std(samples[:, 5])))
    print(u'lc1_s2 = {} \u00B1 {}'.format(lc1_s2, np.std(samples[:, 6])))
    print(u'lc1_f2 = {} \u00B1 {}'.format(lc1_f2, np.std(samples[:, 7])))

    # Plot triangle plot
    fig = corner.corner(samples,
                        labels=["$lc1_l1$",
                                "$lc1_b1$",
                                "$lc1_s1$",
                                "$lc1_f1$",
                                "$lc1_l2$",
                                "$lc1_b2$",
                                "$lc1_s2$",
                                "$lc1_f2$"],
                        truths=initial,
                        plot_contours=False)
    fig.savefig('{}/corner_{}steps_{}walkers.png'.format(outdir, nsteps, nwalkers))
    fig.clf()

    x_model = np.linspace(-0.5, 0.5, 1000)
    f_s = np.sqrt(in_ecc)*np.sin(in_omega*np.pi/180.)
    f_c = np.sqrt(in_ecc)*np.cos(in_omega*np.pi/180.)
    ldcs_1 = [in_ldc_1_1, in_ldc_1_2]
    final_lc_model1 = light_curve_model(t_obs=x_model,
                                        t0=0.0,
                                        period=1.0,
                                        radius_1=in_radius_1,
                                        radius_2=in_radius_2,
                                        sbratio=in_sbratio,
                                        a=in_a,
                                        q=in_q,
                                        incl=in_incl,
                                        f_s=f_s,
                                        f_c=f_c,
                                        ldc_1=ldcs_1,
                                        spots_1=[[lc1_l1], [lc1_b1], [lc1_s1], [lc1_f1]],
                                        spots_2=[[lc1_l2], [lc1_b2], [lc1_s2], [lc1_f2]])

    phase_lc1 = ((x_lc1 - in_t0)/in_period)%1

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.plot(phase_lc1, y_lc1, 'k.')
    ax.plot(phase_lc1-1, y_lc1, 'k.')
    ax.plot(x_model, final_lc_model1, 'r-', lw=2)
    ax.set_xlim(-0.02, 0.02)
    ax.set_ylim(0.96, 1.02)
    ax.set_xlabel('Orbital Phase')
    ax.set_ylabel('Relative Flux')
    fig.savefig('{}/chain_{}steps_{}walkers_fitted_models.png'.format(outdir,
                                                                      nsteps,
                                                                      nwalkers))
