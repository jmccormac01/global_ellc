# -*- coding: latin-1 -*-
"""
ToDo:
    1. Need to generalise how to handle fixed parameters and
       generalise how to handle the priors. Get ideas from
       ellc emcee exmaple
    2. Output the chains so they can be plotted quickly if a
       file exists already
"""
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

# takes in the binary parameters and returns an ellc fit
# for the light curve
def light_curve_model(t_obs, t0, period, radius_1, radius_2,
                      sbratio, incl, f_s, f_c, a, q, ldc_1):
    """
    Add docstring
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
                       f_s=f_s)
    return lc_model

# takes in the binary parameters and returns an ellc fit
# for the radial velocity curve
def rv_curve_model(t_obs, t0, period, radius_1, radius_2,
                   sbratio, incl, f_s, f_c, a, q, v_systemic):
    """
    Add docstring
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
    rv1 = rv1 + v_systemic
    return rv1

def lnprior(theta):
    """
    Needs generalising for priors

    These are set for J234318.41

    Add docstring
    """
    radius_1, radius_2, incl, t0, \
    period, ecc, omega, a, ldc_1_1, ldc_1_2, \
    v_systemic, q = theta
    # uniform priors for the parameters in theta
    if 0.02 <= radius_1 <= 0.04 and \
        0.003 < radius_2 < 0.007 and \
        88 < incl <= 90 and \
        0.0 <= ecc <= 0.2 and \
        30.0 <= a <= 35.0 and \
        75 <= omega < 80 and \
        -15 >= v_systemic >= -25 and \
        0.05 < q < 0.145:
        return 0.0
    else:
        return -np.inf

def lnlike(theta, x_lc, y_lc, yerr_lc, x_rv, y_rv, yerr_rv):
    """
    Add docstring
    """
    # unpack theta and pass parms to model
    radius_1, radius_2, incl, t0, period, \
    ecc, omega, a, ldc_1_1, ldc_1_2, \
    v_systemic, q = theta

    # set the two ldcs into a list for ellc
    ldcs_1 = [ldc_1_1, ldc_1_2]

    # fixed parameters
    sbratio = 0.0

    f_s = np.sqrt(ecc)*np.sin(omega*np.pi/180.)
    f_c = np.sqrt(ecc)*np.cos(omega*np.pi/180.)
    # light curve likelihood function
    model_lc = light_curve_model(t_obs=x_lc,
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
    if True in np.isnan(model_lc) or np.min(model_lc) <= 0:
        lnlike_lc = -np.inf
    else:
        inv_sigma2_lc = 1.0/(yerr_lc**2)
        lc_eq_p1 = (y_lc-model_lc)**2*inv_sigma2_lc - np.log(inv_sigma2_lc)
        lnlike_lc = -0.5*(np.sum(lc_eq_p1) - np.log(len(y_lc) + 1))

    # rv curve likelihood function
    model_rv = rv_curve_model(t_obs=x_rv,
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
                              v_systemic=v_systemic)
    if True in np.isnan(model_rv):
        lnlike_rv = -np.inf
    else:
        inv_sigma2_rv = 1.0/(yerr_rv**2)
        rv_eq_p1 = (y_rv-model_rv)**2*inv_sigma2_rv - np.log(inv_sigma2_rv)
        lnlike_rv = -0.5*(np.sum(rv_eq_p1) - np.log(len(y_rv) + 1))

    # sum to get overall likelihood function
    lnlike = lnlike_lc + lnlike_rv
    return lnlike

def lnprob(theta, x_lc, y_lc, yerr_lc, x_rv, y_rv, yerr_rv):
    """
    Add docstring
    """
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x_lc, y_lc, yerr_lc, x_rv, y_rv, yerr_rv)

if __name__ == "__main__":
    # initial guesses of the parameters
    in_radius_1 = 0.0305      #solar radii
    in_radius_2 = 0.004825    #solar radii
    in_sbratio = 0.0          # fixed = set in lnlike
    in_q = 0.0957             # fixed = set in lnlike
    in_incl = 89.55
    in_t0 = 2453592.73266
    in_period = 16.953598
    in_ecc = 0.1597
    in_omega = 78.418
    in_a = 32.05           #solar radii
    in_ldc_1_1 = 0.446
    in_ldc_1_2 = 0.177
    in_v_systemic = -21.236
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
               in_v_systemic,
               in_q]
    # used in plotting
    parameters = ['r1', 'r2', 'inc', 'T0',
                  'P', 'ecc', 'omega', 'a',
                  'ldc1', 'ldc2', 'v_sys', 'q']
    # set up the weights for the initialisation
    # these weights are used to scattter the walkers
    # if using a prior make sure they are not scattered
    # outside the range of the prior
    weights = [5e-4, 5e-4, 1e-2, 1e-3, 5e-4, 5e-2,
               1e-1, 1e-1, 1e-3, 1e-3, 1e-1, 1e-3]
    # check the lists are the same length
    assert len(initial) == len(parameters) == len(weights)

    # grab the lc and rv data for fitting
    #lc_file = '/Users/jmcc/Dropbox/EBLMS/J23431841/J234318.41_nospots.dat'
    lc_file = '/Users/jmcc/Dropbox/EBLMS/J23431841/NITES_J234318.41_20131010_Clear_F2.lc.txt'
    rv_file = '/Users/jmcc/Dropbox/EBLMS/J23431841/J234318.41_NOT.rv'
    x_lc, y_lc, yerr_lc = np.loadtxt(lc_file, usecols=[2, 3, 4], unpack=True)
    x_rv, y_rv, yerr_rv = np.loadtxt(rv_file, usecols=[0, 1, 2], unpack=True)

    # set up the sampler
    ndim = len(initial)
    nwalkers = 4*len(initial)
    nsteps = 2000
    # set up the starting positions
    pos = [initial + weights*np.random.randn(ndim) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(x_lc, y_lc, yerr_lc,
                                          x_rv, y_rv, yerr_rv))

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
        fig.savefig('chain_{}steps_{}walkers_{}.png'.format(nsteps,
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
    v_systemic = np.median(samples[:, 10])
    q = np.median(samples[:, 11])

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
    print('v_systemic = {} ± {}'.format(v_systemic, np.std(samples[:, 10])))
    print('q = {} ± {}'.format(q, np.std(samples[:, 11])))

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
                                "$v_sys$",
                                "$q$"],
                        truths=initial,
                        plot_contours=False)
    fig.savefig('corner_{}steps_{}walkers.png'.format(nsteps, nwalkers))
    fig.clf()

    # take most likely set of parameters and plot the models
    # make a dense mesh of time points for the lcs and RVs
    # this is done in phase space for simplicity,
    # i.e. P = 1 and T0 = 0.0 in model
    x_model = np.linspace(-0.5, 0.5, 1000)
    x_rv_model = np.linspace(t0, t0+period, 1000)
    f_s = np.sqrt(ecc)*np.sin(omega*np.pi/180.)
    f_c = np.sqrt(ecc)*np.cos(omega*np.pi/180.)
    ldcs_1 = [ldc_1_1, ldc_1_2]
    final_lc_model = light_curve_model(t_obs=x_model,
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
                                    v_systemic=v_systemic)

    phase_lc = ((x_lc - t0)/period)%1
    phase_rv = ((x_rv - t0)/period)%1
    phase_rv_model = ((x_rv_model-t0)/period)%1

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(phase_lc, y_lc, 'k.')
    ax[0].plot(phase_lc-1, y_lc, 'k.')
    ax[0].plot(x_model, final_lc_model, 'r-', lw=2)
    ax[0].set_xlim(-0.02, 0.02)
    ax[0].set_ylim(0.96, 1.02)
    ax[0].set_xlabel('Orbital Phase')
    ax[0].set_ylabel('Relative Flux')
    ax[1].plot(phase_rv, y_rv, 'k.')
    ax[1].plot(phase_rv_model, final_rv_model, 'r-', lw=2)
    ax[1].set_xlim(0, 1)
    ax[1].set_xlabel('Orbital Phase')
    ax[1].set_ylabel('Radial Velocity')
    fig.savefig('chain_{}steps_{}walkers_fitted_models.png'.format(nsteps, nwalkers))
