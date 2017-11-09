# -*- coding: latin-1 -*-
"""
This is an attempt at a multi-instrument version of emcee_combined.py

ToDo:
    1. Need to generalise how to handle fixed parameters and
       generalise how to handle the priors. Get ideas from
       ellc emcee example
    2. Output the chains so they can be plotted quickly if a
       file exists already
    3. Generalise how to spread out the walkers in the start

J234318.41 ToDo:
    1. Remove 1s in partial
    2. Speak to people about bright spot (double check reductions?)
    3. Fix systemic velocity offsets
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
                       grid_1='default',
                       grid_2='default',
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
    radius_1, radius_2, incl, t0, \
    period, ecc, omega, a, ldc_1_1, ldc_1_2, \
    lc1_l1, lc1_b1, lc1_s1, lc1_f1, \
    lc2_l1, lc2_b1, lc2_s1, lc2_f1, \
    v_sys1, v_sys2, v_sys3, q = theta
    # uniform priors for the parameters in theta
    if 0.02 <= radius_1 <= 0.04 and \
        0.002 < radius_2 < 0.007 and \
        88 < incl <= 90 and \
        0.1 <= ecc <= 0.2 and \
        30.0 <= a <= 34.0 and \
        70 <= omega < 90 and \
        1 <= lc1_s1 <= 15 and \
        0 <= lc1_l1 <= 90 and \
        -15 <= lc1_b1 <= 15 and \
        0.0 <= lc1_f1 <= 1.0 and \
        1 <= lc2_s1 <= 15 and \
        0 <= lc2_l1 <= 90 and \
        -15 <= lc2_b1 <= 15 and \
        0.0 <= lc2_f1 <= 1.0 and \
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
           x_lc2, y_lc2, yerr_lc2,
           x_lc3, y_lc3, yerr_lc3,
           x_rv1, y_rv1, yerr_rv1,
           x_rv2, y_rv2, yerr_rv2,
           x_rv3, y_rv3, yerr_rv3):
    """
    Work out the log likelihood for the proposed model
    """
    # unpack theta and pass parms to model
    radius_1, radius_2, incl, t0, period, \
    ecc, omega, a, ldc_1_1, ldc_1_2, \
    lc1_l1, lc1_b1, lc1_s1, lc1_f1, \
    lc2_l1, lc2_b1, lc2_s1, lc2_f1, \
    v_sys1, v_sys2, v_sys3, q = theta

    # set the two ldcs into a list for ellc
    ldcs_1 = [ldc_1_1, ldc_1_2]

    # fixed parameters
    sbratio = 0.0
    f_s = np.sqrt(ecc)*np.sin(omega*np.pi/180.)
    f_c = np.sqrt(ecc)*np.cos(omega*np.pi/180.)

    # light curve 1 likelihood function - spotty
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
                                  ldc_1=ldcs_1,
                                  spots_1=[[lc1_l1], [lc1_b1], [lc1_s1], [lc1_f1]])
    lnlike_lc1 = lnlike_sub('phot', model_lc1, y_lc1, yerr_lc1)

    # light curve 2 likelihood function - spotty
    model_lc2 = light_curve_model(t_obs=x_lc2,
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
                                  ldc_1=ldcs_1,
                                  spots_1=[[lc2_l1], [lc2_b1], [lc2_s1], [lc2_f1]])
    lnlike_lc2 = lnlike_sub('phot', model_lc2, y_lc2, yerr_lc2)

    # light curve 3 likelihood function - non-spotty
    model_lc3 = light_curve_model(t_obs=x_lc3,
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
    lnlike_lc3 = lnlike_sub('phot', model_lc3, y_lc3, yerr_lc3)

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
    lnlike = lnlike_lc1 + lnlike_lc2 + lnlike_lc3 + lnlike_rv1 + lnlike_rv2 + lnlike_rv3
    return lnlike

def lnprob(theta,
           x_lc1, y_lc1, yerr_lc1,
           x_lc2, y_lc2, yerr_lc2,
           x_lc3, y_lc3, yerr_lc3,
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
                       x_lc2, y_lc2, yerr_lc2,
                       x_lc3, y_lc3, yerr_lc3,
                       x_rv1, y_rv1, yerr_rv1,
                       x_rv2, y_rv2, yerr_rv2,
                       x_rv3, y_rv3, yerr_rv3)

if __name__ == "__main__":
    # initial guesses of the parameters
    in_radius_1 = 0.029010     #solar radii
    in_radius_2 = 0.004530     #solar radii
    in_sbratio = 0.0           # fixed = set in lnlike
    in_q = 0.09904
    in_incl = 89.564
    in_t0 = 2453592.72423
    in_period = 16.953634
    in_ecc = 0.1467
    in_omega = 79.98
    in_a = 31.8           #solar radii
    in_ldc_1_1 = 0.4428
    in_ldc_1_2 = 0.1873
    in_lc1_l1 = 31.008 # lc1 spot params from initial fit on its own
    in_lc1_b1 = -4.95
    in_lc1_s1 = 5.954
    in_lc1_f1 = 0.899
    in_lc2_l1 = 33.986 # lc2 spot params from initial fit on its own
    in_lc2_b1 = -5.257
    in_lc2_s1 = 11.033
    in_lc2_f1 = 0.899
    in_v_sys1 = -21.260
    in_v_sys2 = -20.812
    in_v_sys3 = -20.994
    # list of initial guesses
    initial = [in_radius_1,
               in_radius_2,
               in_incl,
               in_t0,
               in_period,
               in_ecc,
               in_omega,
               in_a,
               in_ldc_1_1,
               in_ldc_1_2,
               in_lc1_l1,
               in_lc1_b1,
               in_lc1_s1,
               in_lc1_f1,
               in_lc2_l1,
               in_lc2_b1,
               in_lc2_s1,
               in_lc2_f1,
               in_v_sys1,
               in_v_sys2,
               in_v_sys3,
               in_q]
    # used in plotting
    parameters = ['r1', 'r2', 'inc', 'T0', 'P', 'ecc',
                  'omega', 'a', 'ldc1', 'ldc2',
                  'lc1_l1', 'lc1_b1', 'lc1_s1', 'lc1_f1',
                  'lc2_l1', 'lc2_b1', 'lc2_s1', 'lc2_f1',
                  'v_sys1', 'v_sys2', 'v_sys3', 'q']
    # set up the weights for the initialisation
    # these weights are used to scattter the walkers
    # if using a prior make sure they are not scattered
    # outside the range of the prior
    weights = [1e-4, 1e-4, 1e-2, 1e-3, 1e-4, 1e-3,
               1e-1, 1e-2, 1e-3, 1e-3,
               1e-2, 1e-2, 1e-2, 1e-2,
               1e-2, 1e-2, 1e-2, 1e-2,
               1e-1, 1e-1, 1e-1, 1e-3]
    # check the lists are the same length
    assert len(initial) == len(parameters) == len(weights)

    # READ IN THE DATA
    datadir = '/Users/jmcc/Dropbox/EBLMs/J23431841'
    outdir = '{}/output'.format(datadir)
    # phot
    lc1_file = 'NITES_J234318.41_20120829_Clear_F1.lc.txt'
    lc2_file = 'NITES_J234318.41_20130923_Clear_F2.lc.txt'
    lc3_files = ['NITES_J234318.41_20131010_Clear_F2.lc.txt',
                 'NITES_J234318.41_20141001_Clear_F1.lc.txt']
    x_lc1, y_lc1, yerr_lc1 = np.loadtxt('{}/{}'.format(datadir, lc1_file),
                                        usecols=[2, 3, 4], unpack=True)
    x_lc2, y_lc2, yerr_lc2 = np.loadtxt('{}/{}'.format(datadir, lc2_file),
                                        usecols=[2, 3, 4], unpack=True)
    # read in multiple lc files that are to be treated the same
    # e.g. non-spotty in this case
    x_lc3, y_lc3, yerr_lc3 = [], [], []
    for lc_file in lc3_files:
        x_lc, y_lc, yerr_lc = np.loadtxt('{}/{}'.format(datadir, lc_file),
                                         usecols=[2, 3, 4], unpack=True)
        x_lc3.append(x_lc)
        y_lc3.append(y_lc)
        yerr_lc3.append(yerr_lc)
    # stack the final lcs into one array
    x_lc3 = np.hstack(x_lc3)
    y_lc3 = np.hstack(y_lc3)
    yerr_lc3 = np.hstack(yerr_lc3)

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
    nwalkers = 4*len(initial)
    nsteps = 500
    # set up the starting positions
    pos = [initial + weights*np.random.randn(ndim) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(x_lc1, y_lc1, yerr_lc1,
                                          x_lc2, y_lc2, yerr_lc2,
                                          x_lc3, y_lc3, yerr_lc3,
                                          x_rv1, y_rv1, yerr_rv1,
                                          x_rv2, y_rv2, yerr_rv2,
                                          x_rv3, y_rv3, yerr_rv3))

    # run the production chain
    print("Running MCMC...")
    sampler.run_mcmc(pos, nsteps, rstate0=np.random.get_state())
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
    ldc_1_1 = np.median(samples[:, 8])
    ldc_1_2 = np.median(samples[:, 9])
    lc1_l1 = np.median(samples[:, 10])
    lc1_b1 = np.median(samples[:, 11])
    lc1_s1 = np.median(samples[:, 12])
    lc1_f1 = np.median(samples[:, 13])
    lc2_l1 = np.median(samples[:, 14])
    lc2_b1 = np.median(samples[:, 15])
    lc2_s1 = np.median(samples[:, 16])
    lc2_f1 = np.median(samples[:, 17])
    v_sys1 = np.median(samples[:, 18])
    v_sys2 = np.median(samples[:, 19])
    v_sys3 = np.median(samples[:, 20])
    q = np.median(samples[:, 21])

    print('radius_1 = {} ± {}'.format(radius_1, np.std(samples[:, 0])))
    print('radius_2 = {} ± {}'.format(radius_2, np.std(samples[:, 1])))
    print('incl = {} ± {}'.format(incl, np.std(samples[:, 2])))
    print('t0 = {} ± {}'.format(t0, np.std(samples[:, 3])))
    print('period = {} ± {}'.format(period, np.std(samples[:, 4])))
    print('ecc = {} ± {}'.format(ecc, np.std(samples[:, 5])))
    print('omega = {} ± {}'.format(omega, np.std(samples[:, 6])))
    print('a = {} ± {}'.format(a, np.std(samples[:, 7])))
    print('ldc_1_1 = {} ± {}'.format(ldc_1_1, np.std(samples[:, 8])))
    print('ldc_1_2 = {} ± {}'.format(ldc_1_2, np.std(samples[:, 9])))
    print('lc1_l1 = {} ± {}'.format(lc1_l1, np.std(samples[:, 10])))
    print('lc1_b1 = {} ± {}'.format(lc1_b1, np.std(samples[:, 11])))
    print('lc1_s1 = {} ± {}'.format(lc1_s1, np.std(samples[:, 12])))
    print('lc1_f1 = {} ± {}'.format(lc1_f1, np.std(samples[:, 13])))
    print('lc2_l1 = {} ± {}'.format(lc2_l1, np.std(samples[:, 14])))
    print('lc2_b1 = {} ± {}'.format(lc2_b1, np.std(samples[:, 15])))
    print('lc2_s1 = {} ± {}'.format(lc2_s1, np.std(samples[:, 16])))
    print('lc2_f1 = {} ± {}'.format(lc2_f1, np.std(samples[:, 17])))
    print('v_sys1 = {} ± {}'.format(v_sys1, np.std(samples[:, 18])))
    print('v_sys2 = {} ± {}'.format(v_sys2, np.std(samples[:, 19])))
    print('v_sys3 = {} ± {}'.format(v_sys3, np.std(samples[:, 20])))
    print('q = {} ± {}'.format(q, np.std(samples[:, 21])))

    # Plot triangle plot
    fig = corner.corner(samples,
                        labels=["$radius_1$",
                                "$radius_2$",
                                "$incl$",
                                "$t0$",
                                "$period$",
                                "$ecc$",
                                "$omega$",
                                "$a$",
                                "$ldc1_1$",
                                "$ldc1_2$",
                                "$lc1_l1$",
                                "$lc1_b1$",
                                "$lc1_s1$",
                                "$lc1_f1$",
                                "$lc2_l1$",
                                "$lc2_b1$",
                                "$lc2_s1$",
                                "$lc2_f1$",
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
    ldcs_1 = [ldc_1_1, ldc_1_2]
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
                                        ldc_1=ldcs_1,
                                        spots_1=[[lc1_l1], [lc1_b1], [lc1_s1], [lc1_f1]])
    final_lc_model2 = light_curve_model(t_obs=x_model,
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
                                        ldc_1=ldcs_1,
                                        spots_1=[[lc2_l1], [lc2_b1], [lc2_s1], [lc2_f1]])
    final_lc_model3 = light_curve_model(t_obs=x_model,
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
    phase_lc2 = ((x_lc2 - t0)/period)%1
    phase_lc3 = ((x_lc3 - t0)/period)%1
    phase_rv1 = ((x_rv1 - t0)/period)%1
    phase_rv2 = ((x_rv2 - t0)/period)%1
    phase_rv3 = ((x_rv3 - t0)/period)%1
    phase_rv_model = ((x_rv_model-t0)/period)%1

    fig, ax = plt.subplots(4, 1, figsize=(15, 15))
    ax[0].plot(phase_lc1, y_lc1, 'k.')
    ax[0].plot(phase_lc1-1, y_lc1, 'k.')
    ax[0].plot(x_model, final_lc_model1, 'r-', lw=2)
    ax[0].set_xlim(-0.02, 0.02)
    ax[0].set_ylim(0.96, 1.02)
    ax[0].set_xlabel('Orbital Phase')
    ax[0].set_ylabel('Relative Flux')
    ax[1].plot(phase_lc2, y_lc2, 'k.')
    ax[1].plot(phase_lc2-1, y_lc2, 'k.')
    ax[1].plot(x_model, final_lc_model2, 'r-', lw=2)
    ax[1].set_xlim(-0.02, 0.02)
    ax[1].set_ylim(0.96, 1.02)
    ax[1].set_xlabel('Orbital Phase')
    ax[1].set_ylabel('Relative Flux')
    ax[2].plot(phase_lc3, y_lc3, 'k.')
    ax[2].plot(phase_lc3-1, y_lc3, 'k.')
    ax[2].plot(x_model, final_lc_model3, 'g-', lw=2)
    ax[2].set_xlim(-0.02, 0.02)
    ax[2].set_ylim(0.96, 1.02)
    ax[2].set_xlabel('Orbital Phase')
    ax[2].set_ylabel('Relative Flux')
    ax[3].plot(phase_rv1, y_rv1, 'k.')
    ax[3].plot(phase_rv2, y_rv2, 'g.')
    ax[3].plot(phase_rv3, y_rv3, 'b.')
    ax[3].plot(phase_rv_model, final_rv_model, 'r-', lw=2)
    ax[3].set_xlim(0, 1)
    ax[3].set_xlabel('Orbital Phase')
    ax[3].set_ylabel('Radial Velocity')
    fig.savefig('{}/chain_{}steps_{}walkers_fitted_models.png'.format(outdir,
                                                                      nsteps,
                                                                      nwalkers))
