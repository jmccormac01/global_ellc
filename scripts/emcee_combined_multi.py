"""
This is an attempt at a multi-instrument version of emcee_combined.py
"""
import sys
import argparse as ap
from collections import defaultdict, OrderedDict
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

def argParse():
    """
    Read in the command line arguments

    Parameters
    ----------
    None

    Returns
    -------
    args : argparse object
        object with all command line args parsed

    Raises
    ------
    None
    """
    p = ap.ArgumentParser()
    p.add_argument('config', help='path to config file')
    return p.parse_args()

def readConfig(infile):
    """
    Read in the configuration file

    This contains information on:
        1. data location
        2. light curves and rvs
        3. input parameters with priors

    Parameters
    ----------
    infile : string
        path to the configuration file for a given object

    Returns
    -------
    config : array-like
        object containing all configuration parameters

    Raises
    ------
    None
    """
    config = OrderedDict()
    f = open(infile).readlines()
    for line in f:
        # skip comment lines
        if line.startswith('#'):
            continue
        sp = line.split()
        # look for data locations etc
        if sp[0] == 'data_dir':
            config['data_dir'] = sp[1]
            continue
        elif sp[0] == 'out_dir':
            config['out_dir'] = sp[1]
            continue
        elif sp[0] == 'lc':
            if "lcs" not in config.keys():
                config['lcs'] = defaultdict(list)
            filt = sp[1]
            lc_file = sp[2]
            config['lcs'][filt].append(lc_file)
            continue
        elif sp[0] == 'rv':
            if "rvs" not in config.keys():
                config['rvs'] = defaultdict(list)
            inst = sp[1]
            rv_file = sp[2]
            config['rvs'][inst].append(rv_file)
            continue
        # read Fixed parameters
        if sp[1] == 'F':
            param = sp[0]
            value = float(sp[2])
            if 'fixed' not in config.keys():
                config['fixed'] = OrderedDict()
            config['fixed'][param] = value
        # read fit parameters with No prior
        elif sp[1] == 'N':
            if 'no_prior' not in config.keys():
                config['no_prior'] = OrderedDict()
            param = sp[0]
            value = float(sp[2])
            weight = float(sp[3])
            config['no_prior'][param] = {'value': value,
                                         'weight': weight}
        # read fit parameters with Uniform priors
        elif sp[1] == 'U':
            if 'uniform_prior' not in config.keys():
                config['uniform_prior'] = OrderedDict()
            if sp[0] == 'vsys':
                param = sp[0]
                sys_inst = sp[2]
                value = float(sp[3])
                weight = float(sp[4])
                prior_l = float(sp[5])
                prior_h = float(sp[6])
                if "vsys" not in config['uniform_prior'].keys():
                    config['uniform_prior']['vsys'] = defaultdict(list)
                config['uniform_prior']['vsys'][sys_inst] = {'value': value,
                                                             'weight': weight,
                                                             'prior_l': prior_l,
                                                             'prior_h': prior_h}
            else:
                param = sp[0]
                value = float(sp[2])
                weight = float(sp[3])
                prior_l = float(sp[4])
                prior_h = float(sp[5])
                if 'uniform_prior' not in config.keys():
                    config['uniform_prior'] = OrderedDict()
                config['uniform_prior'][param] = {'value': value,
                                                  'weight': weight,
                                                  'prior_l': prior_l,
                                                  'prior_h': prior_h}
        # eventually read in the Gaussian parameters
    try:
        assert len(config['rvs']) == len(config['uniform_prior']['vsys']), "Mismatching RV + Vsys!"
    except KeyError:
        print('Mismatch in RVs + matching Vsys values, exiting!')
        sys.exit(1)
    return config

def dataLoader(config, data_type):
    """
    Generic data loadng function

    This function works with phot and rvs
    assuming the file format is:
        time  measurment  error

    Parameters
    ----------
    config : array-like
        object containing all configuration parameters
    data_type : string
        type of data to read (e.g. lcs | rvs)

    Returns
    -------
    x_data : array-like
        array of time data
    y_data : array-like
        array of measurement data (phot or rvs)
    yerr_data : array-like
        array of errors on measurments

    Raises
    ------
    None
    """
    x_data = OrderedDict()
    y_data = OrderedDict()
    yerr_data = OrderedDict()
    for filt in config[data_type]:
        x_dat, y_dat, yerr_dat = [], [], []
        for dat in config[data_type][filt]:
            infile = "{}/{}".format(config['data_dir'], dat)
            x, y, e = np.loadtxt(infile, usecols=[0, 1, 2], unpack=True)
            x_dat.append(x)
            y_dat.append(y)
            yerr_dat.append(e)
        # stack the light curves into the global lc holder
        x_data[filt] = np.hstack(x_dat)
        y_data[filt] = np.hstack(y_dat)
        yerr_data[filt] = np.hstack(yerr_dat)
    return x_data, y_data, yerr_data

def light_curve_model(t_obs, t0, period, radius_1, radius_2,
                      sbratio, incl, f_s, f_c, a, q, ldc_1,
                      spots_1=None, spots_2=None):
    """
    Takes in the binary parameters and returns an ellc model
    light curve. This can be used to generate models during
    the fitting process, or the final model when the parameters
    have been deteremined

    Parameters
    ----------
    t_obs : array-like
        array of times of observation
    t0 : float
        epoch of eclipsing system
    period : float
        orbital period of binary
    radius_1 : float
        radius of the primary component in units of a (r1/a)
    radius_2 : float
        radius of the secondary component in units of a (r2/a)
    sbratio : float
        surface brightness ratio between component 1 and 2
    incl : float
        inclination of binary orbit
    f_s : float
    f_c : float
    a : float
        semi-major axis of binary in units of r1 (a/r1)
    q : float
        mass ratio of the binary (m2/m1)
    ldc_1 : array-like
        LDCs for primary eclipse ([ldc_1_1, ldc_1_2], assumes quadratic law)
    spots_1 : array-like
        spot parameters for spot_1 [check order!]
    spots_2 : array-like
        spot parameters for spot_2 [check order!]

    Returns
    -------
    lc_model : array-like
        model of the binary using input parameters

    Raises
    ------
    None
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

def lnprior(theta, config, n_priors):
    """
    Add docstring
    """
    priors = config['uniform_prior']
    for i, p in enumerate(priors):
        if p != 'vsys':
            val = theta[i]
            llim = priors[p]['prior_l']
            ulim = priors[p]['prior_h']
            # check for incorrect priors
            if llim > ulim:
                print('{} priors wrong! {} > {}'.format(p, llim, ulim))
                sys.exit()
            if val < llim or val > ulim:
                return -np.inf
    imax = i
    # there are still some values to check, hence vsys values
    if imax < n_priors:
        for j, p in enumerate(priors['vsys']):
            val = theta[imax+j]
            llim = priors['vsys'][p]['prior_l']
            ulim = priors['vsys'][p]['prior_h']
            # check for incorrect priors
            if llim > ulim:
                print('{} priors wrong! {} > {}'.format(p, llim, ulim))
                sys.exit()
            if val < llim or val > ulim:
                return -np.inf
    return 0.0

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

def lnlike(theta, config,
           x_lc, y_lc, yerr_lc,
           x_rv, y_rv, yerr_rv):
    """
    Work out the log likelihood for the proposed model
    """
    # make copies to make coding next bit easier
    params = config['parameters']
    fixed = config['fixed']
    no_prior = config['no_prior']
    uniform = config['uniform_prior']
    # t0
    if 't0' in no_prior or 't0' in uniform:
        t0 = theta[params.index('t0')]
    elif 't0' in fixed:
        t0 = fixed['t0']
    else:
        raise IndexError('Cannot find t0 in lnlike')
    # period
    if 'period' in no_prior or 'period' in uniform:
        period = theta[params.index('period')]
    elif 'period' in fixed:
        period = fixed['period']
    else:
        raise IndexError('Cannot find period in lnlike')
    # radius_1
    if 'r1_a' in no_prior or 'r1_a' in uniform:
        r1_a = theta[params.index('r1_a')]
    elif 'r1_a' in fixed:
        r1_a = fixed['r1_a']
    else:
        raise IndexError('Cannot find r1_a in lnlike')
    # radius_2
    if 'r2_a' in no_prior or 'r2_a' in uniform:
        r2_a = theta[params.index('r2_a')]
    elif 'r2_a' in fixed:
        r2_a = fixed['r2_a']
    else:
        raise IndexError('Cannot find r2_a in lnlike')
    # sbratio
    if 'sbratio' in no_prior or 'sbratio' in uniform:
        sbratio = theta[params.index('sbratio')]
    elif 'sbratio' in fixed:
        sbratio = fixed['sbratio']
    else:
        raise IndexError('Cannot find sbratio in lnlike')
    # a_Rs
    if 'a_r1' in no_prior or 'a_r1' in uniform:
        a_r1 = theta[params.index('a_r1')]
    elif 'a_r1' in fixed:
        a_r1 = fixed['a_r1']
    else:
        raise IndexError('Cannot find a_r1 in lnlike')
    # q
    if 'q' in no_prior or 'q' in uniform:
        q = theta[params.index('q')]
    elif 'q' in fixed:
        q = fixed['q']
    else:
        raise IndexError('Cannot find q in lnlike')
    # incl
    if 'incl' in no_prior or 'incl' in uniform:
        incl = theta[params.index('incl')]
    elif 'incl' in fixed:
        incl = fixed['incl']
    else:
        raise IndexError('Cannot find incl in lnlike')
    # ecc
    if 'ecc' in no_prior or 'ecc' in uniform:
        ecc = theta[params.index('ecc')]
    elif 'ecc' in fixed:
        ecc = fixed['ecc']
    else:
        raise IndexError('Cannot find ecc in lnlike')
    # omega
    if 'omega' in no_prior or 'omega' in uniform:
        omega = theta[params.index('omega')]
    elif 'omega' in fixed:
        omega = fixed['omega']
    else:
        raise IndexError('Cannot find omega in lnlike')
    # ldc_1_1
    if 'ldc_1_1' in no_prior or 'ldc_1_1' in uniform:
        ldc_1_1 = theta[params.index('ldc_1_1')]
    elif 'ldc_1_1' in fixed:
        ldc_1_1 = fixed['ldc_1_1']
    else:
        raise IndexError('Cannot find ldc_1_1 in lnlike')
    # ldc_1_2
    if 'ldc_1_2' in no_prior or 'ldc_1_2' in uniform:
        ldc_1_2 = theta[params.index('ldc_1_2')]
    elif 'ldc_1_2' in fixed:
        ldc_1_2 = fixed['ldc_1_2']
    else:
        raise IndexError('Cannot find ldc_1_2 in lnlike')
    # tweaking parameters
    ldcs_1 = [ldc_1_1, ldc_1_2]
    f_s = np.sqrt(ecc)*np.sin(omega*np.pi/180.)
    f_c = np.sqrt(ecc)*np.cos(omega*np.pi/180.)

    # calculate lnlike of light curves
    lnlike_lc = 0.0
    for filt in x_lc:
        model_lc = light_curve_model(t_obs=x_lc[filt],
                                     t0=t0,
                                     period=period,
                                     radius_1=r1_a,
                                     radius_2=r2_a,
                                     sbratio=sbratio,
                                     a=a_r1,
                                     q=q,
                                     incl=incl,
                                     f_s=f_s,
                                     f_c=f_c,
                                     ldc_1=ldcs_1)
        lnlike_lc += lnlike_sub('phot', model_lc, y_lc[filt], yerr_lc[filt])

    # calculate lnlike of the radial velocities
    lnlike_rv = 0.0
    for inst in x_rv:
        vsys = theta[params.index('vsys_{}'.format(inst))]
        model_rv = rv_curve_model(t_obs=x_rv[inst],
                                  t0=t0,
                                  period=period,
                                  radius_1=r1_a,
                                  radius_2=r2_a,
                                  sbratio=sbratio,
                                  a=a_r1,
                                  q=q,
                                  incl=incl,
                                  f_s=f_s,
                                  f_c=f_c,
                                  v_sys=vsys)
        lnlike_rv += lnlike_sub('rv', model_rv, y_rv[inst], yerr_rv[inst])

    # sum to get overall likelihood function
    lnlike = lnlike_lc + lnlike_rv
    return lnlike

def lnprob(theta, config, n_priors,
           x_lc, y_lc, yerr_lc,
           x_rv, y_rv, yerr_rv):
    """
    Add docstring
    """
    lp = lnprior(theta, config, n_priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, config,
                       x_lc, y_lc, yerr_lc,
                       x_rv, y_rv, yerr_rv)

if __name__ == "__main__":
    args = argParse()
    config = readConfig(args.config)
    outdir = config['out_dir']
    # setting up the initial, parameters and weights lists
    # these are done as follows:
    #   uniform_priors first (forming the start of theta)
    #   vsys entries, which may not exist
    #   no_priors last, looping over theta and counting n_priors will skip these
    initial = [config['uniform_prior'][c]['value'] for c in config['uniform_prior'] if c != "vsys"]
    parameters = [p for p in config['uniform_prior'] if p != "vsys"]
    weights = [config['uniform_prior'][c]['weight'] for c in config['uniform_prior'] if c != "vsys"]
    # add the different systemic velocities
    if 'vsys' in config['uniform_prior'].keys():
        initial = initial + \
                  [config['uniform_prior']['vsys'][c]['value'] for c in config['uniform_prior']['vsys']]
        parameters = parameters + \
                     ["vsys_"+c for c in config['uniform_prior']['vsys']]
        weights = weights + \
                  [config['uniform_prior']['vsys'][c]['weight'] for c in config['uniform_prior']['vsys']]
    n_priors = len(initial)
    # now add the no priors on to the end
    initial = initial + [config['no_prior'][c]['value'] for c in config['no_prior']]
    parameters = parameters + [p for p in config['no_prior']]
    weights = weights + [config['no_prior'][c]['weight'] for c in config['no_prior']]
    # double check that the assignments have worked ok
    assert len(initial) == len(parameters) == len(weights), "intial != parameters != weights!!"
    # add the initial, parameters, weights to config
    config['initial'] = initial
    config['parameters'] = parameters
    config['weights'] = weights
    # READ IN THE DATA
    x_lc, y_lc, yerr_lc = dataLoader(config, 'lcs')
    x_rv, y_rv, yerr_rv = dataLoader(config, 'rvs')
    # set up the sampler
    ndim = len(initial)
    nwalkers = 4*len(initial)#*8
    nsteps = 100
    # set up the starting positions
    pos = [initial + weights*np.random.randn(ndim) for i in range(nwalkers)]
    # set up the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(config, n_priors,
                                          x_lc, y_lc, yerr_lc,
                                          x_rv, y_rv, yerr_rv))
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
    # calculate the most likely set of parameters
    burnin = int(raw_input('Enter burnin period: '))
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    # determine the most likely set of parameters
    best_params = OrderedDict()
    for i, param in enumerate(config['parameters']):
        best_params[param] = {'value': np.median(samples[:, i]),
                              'error': np.std(samples[:, i])}

    # TODO: print a summary of the best parameters/log it to a file

    # make a corner plot
    labels = ["$"+p+"$" for p in config['parameters']]
    fig = corner.corner(samples,
                        labels=parameters,
                        truths=initial,
                        plot_contours=False)
    fig.savefig('{}/corner_{}steps_{}walkers.png'.format(outdir, nsteps, nwalkers))
    fig.clf()

    # PAUSE for now until I have time to add plotting of final model!
    # TODO: Finish addding plotting of final model
    sys.exit()

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
