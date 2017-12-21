# -*- coding: latin-1 -*-
"""
This is an attempt at a multi-instrument version of emcee_combined.py

ToDo:
    1. Need to generalise how to handle fixed parameters and
       generalise how to handle the priors. Get ideas from
       ellc emcee example
    2. Generalise the handling of multiple instruments
    4. Generalise how to spread out the walkers in the start
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
                       grid_1='sparse', # set these to default again later
                       grid_2='sparse',
                       f_c=f_c,
                       f_s=f_s,
                       spots_1=spots_1,
                       spots_2=spots_2)
    return lc_model

def rv_curve_model(t_obs, t0, period, radius_1, radius_2,
                   sbratio, incl, f_s, f_c, a, q, v_sys):
    """
    Takes in the binary parameters and returns an ellc model
    for the radial velocity curve
    """
    rv1, _ = ellc.rv(t_obs=t_obs,
                     t_zero=t0,
                     period=period,
                     radius_1=radius_1,
                     radius_2=radius_2,
                     incl=incl,
                     sbratio=sbratio,
                     a=a,
                     q=q,
                     shape_1='sphere',
                     shape_2='sphere',
                     grid_1='default',
                     grid_2='default',
                     f_c=f_c,
                     f_s=f_s)
    # account for the systemic
    rv1 = rv1 + v_sys
    return rv1

def lnprior(theta):
    """
    Needs generalising for priors

    These are set for J234318.41

    Add docstring
    """
    # floating ldcs
    #radius_1, radius_2, incl, t0, \
    #period, ecc, omega, a, ldc_1_1, ldc_1_2, \
    #v_sys1, v_sys2, v_sys3, q = theta

    # fixed ldcs
    radius_1, radius_2, incl, t0, \
    period, ecc, omega, a, v_sys1, v_sys2, \
    v_sys3, q = theta

    # uniform priors for the parameters in theta
    if 0.02 <= radius_1 <= 0.04 and \
        0.002 < radius_2 < 0.007 and \
        88 < incl <= 90 and \
        0.1 <= ecc <= 0.2 and \
        28.0 <= a <= 36.0 and \
        70 <= omega < 90 and \
        -15 >= v_sys1 >= -25 and \
        -15 >= v_sys2 >= -25 and \
        -15 >= v_sys3 >= -25 and \
        0.05 < q < 0.145:
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

def lnlike(theta,
           x_lc1, y_lc1, yerr_lc1,
           x_rv1, y_rv1, yerr_rv1,
           x_rv2, y_rv2, yerr_rv2,
           x_rv3, y_rv3, yerr_rv3):
    """
    Work out the log likelihood for the proposed model
    """
    # unpack theta and pass parms to model

    # floating ldcs
    #radius_1, radius_2, incl, t0, period, \
    #ecc, omega, a, ldc_1_1, ldc_1_2, \
    #v_sys1, v_sys2, v_sys3, q = theta

    # set the two ldcs into a list for ellc
    #ldcs_1 = [ldc_1_1, ldc_1_2]

    # fixed ldcs
    radius_1, radius_2, incl, t0, period, \
    ecc, omega, a, v_sys1, v_sys2, v_sys3, q = theta

    # fixed parameters
    ldcs_1 = [in_ldc_1_1, in_ldc_1_2]
    sbratio = in_sbratio
    f_s = np.sqrt(ecc)*np.sin(omega*np.pi/180.)
    f_c = np.sqrt(ecc)*np.cos(omega*np.pi/180.)

    # light curve 3 likelihood function - non-spotty
    model_lc1 = light_curve_model(t_obs=x_lc1,
                                  t0=t0,
                                  period=period,
                                  radius_1=radius_1,
                                  radius_2=radius_2,
                                  sbratio=sbratio,
                                  a=a,
                                  q=q,
                                  incl=incl,
                                  f_s=f_s,
                                  f_c=f_c,
                                  ldc_1=ldcs_1)
    lnlike_lc1 = lnlike_sub('phot', model_lc1, y_lc1, yerr_lc1)

    # rv curve likelihood function for instrument 1
    model_rv1 = rv_curve_model(t_obs=x_rv1,
                               t0=t0,
                               period=period,
                               radius_1=radius_1,
                               radius_2=radius_2,
                               sbratio=sbratio,
                               a=a,
                               q=q,
                               incl=incl,
                               f_s=f_s,
                               f_c=f_c,
                               v_sys=v_sys1)
    lnlike_rv1 = lnlike_sub('rv', model_rv1, y_rv1, yerr_rv1)

    # rv curve likelihood function for instrument 2
    model_rv2 = rv_curve_model(t_obs=x_rv2,
                               t0=t0,
                               period=period,
                               radius_1=radius_1,
                               radius_2=radius_2,
                               sbratio=sbratio,
                               a=a,
                               q=q,
                               incl=incl,
                               f_s=f_s,
                               f_c=f_c,
                               v_sys=v_sys2)
    lnlike_rv2 = lnlike_sub('rv', model_rv2, y_rv2, yerr_rv2)

    # rv curve likelihood function for instrument 3
    model_rv3 = rv_curve_model(t_obs=x_rv3,
                               t0=t0,
                               period=period,
                               radius_1=radius_1,
                               radius_2=radius_2,
                               sbratio=sbratio,
                               a=a,
                               q=q,
                               incl=incl,
                               f_s=f_s,
                               f_c=f_c,
                               v_sys=v_sys3)
    lnlike_rv3 = lnlike_sub('rv', model_rv3, y_rv3, yerr_rv3)

    # sum to get overall likelihood function
    lnlike = lnlike_lc1 + lnlike_rv1 + lnlike_rv2 + lnlike_rv3
    return lnlike

def lnprob(theta,
           x_lc1, y_lc1, yerr_lc1,
           x_rv1, y_rv1, yerr_rv1,
           x_rv2, y_rv2, yerr_rv2,
           x_rv3, y_rv3, yerr_rv3):
    """
    Add docstring
    """
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta,
                       x_lc1, y_lc1, yerr_lc1,
                       x_rv1, y_rv1, yerr_rv1,
                       x_rv2, y_rv2, yerr_rv2,
                       x_rv3, y_rv3, yerr_rv3)

if __name__ == "__main__":
    # initial guesses of the parameters
    in_sbratio = 0.0           # fixed = set in lnlike
    in_radius_1 = 0.029363    #solar radii
    in_radius_2 = 0.004665     #solar radii
    in_incl = 89.6232
    in_t0 = 2453592.74192
    in_period = 16.9535452
    in_ecc = 0.16035
    in_omega = 78.39513
    in_a = 31.650747           #solar radii
    in_ldc_1_1 = 0.3897
    in_ldc_1_2 = 0.1477
    in_v_sys1 = -21.133
    in_v_sys2 = -21.122
    in_v_sys3 = -20.896
    in_q = 0.09649
    # list of initial guesses
    #           in_ldc_1_1,
    #           in_ldc_1_2,
    initial = [in_radius_1,
               in_radius_2,
               in_incl,
               in_t0,
               in_period,
               in_ecc,
               in_omega,
               in_a,
               in_v_sys1,
               in_v_sys2,
               in_v_sys3,
               in_q]
    # used in plotting - with floating ldcs
    #parameters = ['r1', 'r2', 'inc', 'T0', 'P', 'ecc',
    #              'omega', 'a', 'ldc1', 'ldc2',
    #              'v_sys1', 'v_sys2', 'v_sys3', 'q']
    # used in plotting - with fixed ldcs
    parameters = ['r1', 'r2', 'inc', 'T0', 'P', 'ecc',
                  'omega', 'a', 'v_sys1', 'v_sys2',
                  'v_sys3', 'q']
    # set up the weights for the initialisation
    # these weights are used to scattter the walkers
    # if using a prior make sure they are not scattered
    # outside the range of the prior

    # floating ldcs
    #weights = [1e-4, 1e-4, 1e-2, 1e-3, 1e-4, 1e-3,
    #           1e-1, 1e-2, 1e-3, 1e-3,
    #           1e-1, 1e-1, 1e-1, 1e-3]
    # fixed ldcs
    weights = [1e-4, 1e-4, 1e-2, 1e-3, 1e-4, 1e-3,
               1e-1, 1e-2, 1e-1, 1e-1, 1e-1, 1e-3]
    # check the lists are the same length
    assert len(initial) == len(parameters) == len(weights)

    # READ IN THE DATA
    datadir = '/Users/jmcc/Dropbox/EBLMs/J23431841'
    outdir = '{}/output'.format(datadir)
    # phot
    lc1_files = ['NITES_J234318.41_Clear_20120829_F1_A14.lc.txt',
                 'NITES_J234318.41_Clear_20130923_F2_A14.lc.txt',
                 'NITES_J234318.41_Clear_20131010_F1_A14.lc.txt',
                 'NITES_J234318.41_Clear_20141001_F1_A14.lc.txt']
    # read in multiple lc files that are to be treated the same
    # e.g. non-spotty in this case
    x_lc1, y_lc1, yerr_lc1 = [], [], []
    for lc_file in lc1_files:
        x_lc, y_lc, yerr_lc = np.loadtxt('{}/{}'.format(datadir, lc_file),
                                         usecols=[2, 3, 4], unpack=True)
        x_lc1.append(x_lc)
        y_lc1.append(y_lc)
        yerr_lc1.append(yerr_lc)
    # stack the final lcs into one array
    x_lc1 = np.hstack(x_lc1)
    y_lc1 = np.hstack(y_lc1)
    yerr_lc1 = np.hstack(yerr_lc1)

    # RVs
    rv1_file = 'J234318.41_NOT.rv'
    rv2_file = 'J234318.41_SOPHIE.rv'
    rv3_file = 'J234318.41_PARAS.rv'
    # read in possible multiple rv files
    x_rv1, y_rv1, yerr_rv1 = np.loadtxt('{}/{}'.format(datadir, rv1_file),
                                        usecols=[0, 1, 2], unpack=True)
    x_rv2, y_rv2, yerr_rv2 = np.loadtxt('{}/{}'.format(datadir, rv2_file),
                                        usecols=[0, 1, 2], unpack=True)
    x_rv3, y_rv3, yerr_rv3 = np.loadtxt('{}/{}'.format(datadir, rv3_file),
                                        usecols=[0, 1, 2], unpack=True)
    # set up the sampler
    ndim = len(initial)
    nwalkers = 4*len(initial)*8
    nsteps = 1000
    # set up the starting positions
    pos = [initial + weights*np.random.randn(ndim) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(x_lc1, y_lc1, yerr_lc1,
                                          x_rv1, y_rv1, yerr_rv1,
                                          x_rv2, y_rv2, yerr_rv2,
                                          x_rv3, y_rv3, yerr_rv3))

    # run the production chain
    print("Running MCMC...")
    sampler.run_mcmc(pos, nsteps, rstate0=np.random.get_state())
    print("Saving chain...")
    np.savetxt('{}/chain_{}steps_{}walkers.csv'.format(outdir, nsteps, nwalkers),
               np.c_[sampler.chain.reshape((-1, ndim))],
               delimiter=',',header=','.join(parameters))
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
    radius_1 = np.median(samples[:, 0])
    radius_2 = np.median(samples[:, 1])
    incl = np.median(samples[:, 2])
    t0 = np.median(samples[:, 3])
    period = np.median(samples[:, 4])
    ecc = np.median(samples[:, 5])
    omega = np.median(samples[:, 6])
    a = np.median(samples[:, 7])
    #ldc_1_1 = np.median(samples[:, 8])
    #ldc_1_2 = np.median(samples[:, 9])
    v_sys1 = np.median(samples[:, 8])
    v_sys2 = np.median(samples[:, 9])
    v_sys3 = np.median(samples[:, 10])
    q = np.median(samples[:, 11])

    # correc the systematic velocities to the first one
    v_sys2_diff = v_sys1 - v_sys2
    v_sys3_diff = v_sys1 - v_sys3

    print('radius_1 = {} ± {}'.format(radius_1, np.std(samples[:, 0])))
    print('radius_2 = {} ± {}'.format(radius_2, np.std(samples[:, 1])))
    print('incl = {} ± {}'.format(incl, np.std(samples[:, 2])))
    print('t0 = {} ± {}'.format(t0, np.std(samples[:, 3])))
    print('period = {} ± {}'.format(period, np.std(samples[:, 4])))
    print('ecc = {} ± {}'.format(ecc, np.std(samples[:, 5])))
    print('omega = {} ± {}'.format(omega, np.std(samples[:, 6])))
    print('a = {} ± {}'.format(a, np.std(samples[:, 7])))
    #print('ldc_1_1 = {} ± {}'.format(ldc_1_1, np.std(samples[:, 8])))
    #print('ldc_1_2 = {} ± {}'.format(ldc_1_2, np.std(samples[:, 9])))
    print('v_sys1 = {} ± {}'.format(v_sys1, np.std(samples[:, 8])))
    print('v_sys2 = {} ± {}'.format(v_sys2, np.std(samples[:, 9])))
    print('v_sys2_diff = {}'.format(v_sys2_diff))
    print('v_sys3 = {} ± {}'.format(v_sys3, np.std(samples[:, 10])))
    print('v_sys3_diff = {}'.format(v_sys3_diff))
    print('q = {} ± {}'.format(q, np.std(samples[:, 11])))

    # Plot triangle plot
    #                            "$ldc1_1$",
    #                            "$ldc1_2$",
    fig = corner.corner(samples,
                        labels=["$radius_1$",
                                "$radius_2$",
                                "$incl$",
                                "$t0$",
                                "$period$",
                                "$ecc$",
                                "$omega$",
                                "$a$",
                                "$v_sys1$",
                                "$v_sys2$",
                                "$v_sys3$",
                                "$q$"],
                        truths=initial,
                        plot_contours=False)
    fig.savefig('{}/corner_{}steps_{}walkers.png'.format(outdir, nsteps, nwalkers))
    fig.clf()

    # take most likely set of parameters and plot the models
    # make a dense mesh of time points for the lcs and RVs
    # this is done in phase space for simplicity,
    # i.e. P = 1 and T0 = 0.0 in model

    # THIS NEEDS WORK TO CHECK THE FINAL COMBINATION OF DATA + MODEL IS CORRECT
    x_model = np.linspace(-0.5, 0.5, 1000)
    x_rv_model = np.linspace(t0, t0+period, 1000)
    f_s = np.sqrt(ecc)*np.sin(omega*np.pi/180.)
    f_c = np.sqrt(ecc)*np.cos(omega*np.pi/180.)

    # floating ldcs
    #ldcs_1 = [ldc_1_1, ldc_1_2]
    # fixed ldcs
    ldcs_1 = [in_ldc_1_1, in_ldc_1_2]

    # final models
    final_lc_model1 = light_curve_model(t_obs=x_model,
                                        t0=0.0,
                                        period=1.0,
                                        radius_1=radius_1,
                                        radius_2=radius_2,
                                        sbratio=in_sbratio,
                                        a=a,
                                        q=q,
                                        incl=incl,
                                        f_s=f_s,
                                        f_c=f_c,
                                        ldc_1=ldcs_1)
    final_rv_model = rv_curve_model(t_obs=x_rv_model,
                                    t0=t0,
                                    period=period,
                                    radius_1=radius_1,
                                    radius_2=radius_2,
                                    sbratio=in_sbratio,
                                    a=a,
                                    q=q,
                                    incl=incl,
                                    f_s=f_s,
                                    f_c=f_c,
                                    v_sys=v_sys1)

    phase_lc1 = ((x_lc1 - t0)/period)%1
    phase_rv1 = ((x_rv1 - t0)/period)%1
    phase_rv2 = ((x_rv2 - t0)/period)%1
    phase_rv3 = ((x_rv3 - t0)/period)%1
    phase_rv_model = ((x_rv_model-t0)/period)%1

    fig, ax = plt.subplots(2, 1, figsize=(15, 15))
    ax[0].plot(phase_lc1, y_lc1, 'k.')
    ax[0].plot(phase_lc1-1, y_lc1, 'k.')
    ax[0].plot(x_model, final_lc_model1, 'g-', lw=2)
    ax[0].set_xlim(-0.02, 0.02)
    ax[0].set_ylim(0.96, 1.02)
    ax[0].set_xlabel('Orbital Phase')
    ax[0].set_ylabel('Relative Flux')
    ax[1].plot(phase_rv1, y_rv1, 'k.')
    ax[1].plot(phase_rv2, y_rv2 + v_sys2_diff, 'g.')
    ax[1].plot(phase_rv3, y_rv3 + v_sys3_diff, 'b.')
    ax[1].plot(phase_rv_model, final_rv_model, 'r-', lw=2)
    ax[1].set_xlim(0, 1)
    ax[1].set_xlabel('Orbital Phase')
    ax[1].set_ylabel('Radial Velocity')
    fig.savefig('{}/chain_{}steps_{}walkers_fitted_models.png'.format(outdir,
                                                                      nsteps,
                                                                      nwalkers))
