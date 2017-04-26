import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import emcee
import corner
import ellc

# pylint: disable=invalid-name
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
                   sbratio, incl, f_s, f_c, a, q):
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
    return rv1

def lnprior(theta):
    """
    Needs generalising for priors

    These are set for J234318.41
    """
    radius_1, radius_2, incl, t0, \
    period, ecc, omega, a, ldc_1_1, ldc_1_2 = theta
    # uniform priors for the parameters in theta
    if 0.0025 < radius_1 < 0.05 and \
        0.00035 < radius_2 < 0.004 and \
        80 < incl <= 90 and \
        ecc > 0 and \
        0 <= omega < 360:
        return 0.0
    else:
        return -np.inf

def lnlike(theta, x_lc, y_lc, yerr_lc, x_rv, y_rv, yerr_rv):
    # unpack theta and pass parms to model
    radius_1, radius_2, incl, t0, period, \
        ecc, omega, a, ldc_1_1, ldc_1_2 = theta

    # set the two ldcs into a list for ellc
    ldcs_1 = [ldc_1_1, ldc_1_2]

    # fixed parameters
    sbratio = 0.0
    q = 0.1124

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
                              f_c=f_c)
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
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x_lc, y_lc, yerr_lc, x_rv, y_rv, yerr_rv)

if __name__ == "__main__":
    # initial guesses of the parameters
    in_radius_1 = 0.0313     #solar radii
    in_radius_2 = 0.00465    #solar radii
    in_sbratio = 0.0         # fixed = set in lnlike
    in_q = 0.1124            # fixed = set in lnlike
    in_incl = 89.0
    in_t0 = 2456457.89050760
    in_period = 16.95352694
    in_ecc = 0.1
    in_omega = 77.0
    in_a = 31.978            #solar radii
    in_ldc_1_1 = 0.1
    in_ldc_1_2 = 0.3
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
               in_ldc_1_2]
    # used in plotting
    parameters = ['r1', 'r2', 'inc', 'T0',
                  'P', 'ecc', 'omega', 'a',
                  'ldc1', 'ldc2']
    # set up the weights for the initialisation
    weights = [5e-4, 5e-4, 2e-4, 1e-4, 5e-4, 5e-4, 1e-4, 1e-4, 1e-4, 1e-4]
    # check the lists are the same length
    assert len(initial) == len(parameters) == len(weights)

    ############
    ### MCMC ###
    ############

    # grab the lc and rv data for fitting
    lc_file = '/Users/jmcc/Dropbox/EBLMS/J23431841/J234318.41_nospots.dat'
    rv_file = '/Users/jmcc/Dropbox/EBLMS/J23431841/J234318.41_NOT.rv'
    x_lc, y_lc, yerr_lc = np.loadtxt(lc_file, usecols=[0, 1, 2], unpack=True)
    x_rv, y_rv, yerr_rv = np.loadtxt(rv_file, usecols=[0, 1, 2], unpack=True)

    # set up the sampler
    ndim = len(initial)
    nwalkers = 3*len(initial)
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
    # get user to input the burnin period, after they
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

    print('radius_1 = {} ± {}'.format(radius_1, np.std(samples[:, 0])))
    print('radius_2 = {} ± {}'.format(radius_2, np.std(samples[:, 1])))
    print('incl = {} ± {}'.format(incl, np.std(samples[:, 2])))
    print('t0 = {} ± {}'.format(t0, np.std(samples[:, 3])))
    print('period = {} ± {}'.format(period, np.std(samples[:, 4])))
    print('ecc = {} ± {}'.format(ecc, np.std(samples[:, 5])))
    print('omega = {} ± {}'.format(omega, np.std(samples[:, 6])))
    print('a = {} ± {}'.format(a, np.std(samples[:, 7])))
    print('ldc_1_1 = {} ± {}'.format(ldc_1, np.std(samples[:, 8])))
    print('ldc_1_2 = {} ± {}'.format(ldc_1, np.std(samples[:, 9])))

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
                                "$ldc_1_1$",
                                "$ldc_1_2$"],
                        truths=initial,
                        plot_contours=False)
    fig.savefig('corner_{}steps_{}walkers.png'.format(nsteps, nwalkers))
    fig.clf()

    # test this plotting stuff later when mcmc runs ok
    # FUNCTION CALLS BELOW HERE NEED FIXING!!!!
    cont = 0
    if cont > 0:
        ##############################################################
        ### Take most likely set of parameters and plot the models ###
        ##############################################################

        # prep for plotting
        t_model = np.linspace(1605, 1635, 10000)

        # calculate final models
        lc_model = light_curve_model(x_lc, t0, period, radius_1,
                                     radius_2, sbratio, incl, f_c,
                                     f_s, ldc_1)
        rv_model = rv_curve_model(t_model, t0, period, f_c, f_s, a)

        # phase the original flattened data using the calculated t0 and period values
        Phase, Phase_index, bin_phases, bin_means, \
        bin_error = phase_curve(planet='KOI-1741', t0=t0, period=period, step_size=2)

        # subplot 1 - For the light curve
        fig = plt.figure()
        gs = gridspec.GridSpec(3, 3)

        ax1 = fig.add_subplot(gs[0, 0:])
        plt.errorbar(bin_phases, bin_means, yerr=bin_error, fmt='r.')
        plt.plot(Phase, lc_model[Phase_index], 'b')
        plt.errorbar(bin_phases-1, bin_means, yerr=bin_error, fmt='r.')
        plt.plot(Phase-1, lc_model[Phase_index], 'b')

        plt.xlabel('Phase')
        plt.ylabel('Relative Flux')

        # subplot 2 - For the primary eclipse (zoom in)
        ax2 = fig.add_subplot(gs[1, 0])
        plt.errorbar(bin_phases, bin_means, yerr=bin_error, fmt='r.')
        plt.plot(Phase, lc_model[Phase_index], 'b')
        plt.errorbar(bin_phases-1, bin_means, yerr=bin_error, fmt='r.')
        plt.plot(Phase-1, lc_model[Phase_index], 'b')

        plt.xlabel('Phase')
        plt.ylabel('Relative Flux')

        # subplot 3 - For the secondary eclipse (zoom in)
        ax3 = fig.add_subplot(gs[2, 0])
        plt.errorbar(bin_phases, bin_means, yerr=bin_error, fmt='r.')
        plt.plot(Phase, lc_model[Phase_index], 'b')
        plt.errorbar(bin_phases-1, bin_means, yerr=bin_error, fmt='r.')
        plt.plot(Phase-1, lc_model[Phase_index], 'b')

        plt.xlabel('Phase')
        plt.ylabel('Relative Flux')

        # subplot 4 - For the radial velocities
        ax4 = fig.add_subplot(gs[1:, 1:])
        plt.plot(t_model, rv1, 'b')
        plt.errorbar(x_rv, y_rv, yerr=yerr_rv, fmt='r.')

        plt.xlabel('Time [BJD-2454833]')
        plt.ylabel('rv1 [km/s]')

        plt.savefig('chain_{}steps_{}walkers_models.png'.format(nsteps, nwalkers))
        fig.clf()
