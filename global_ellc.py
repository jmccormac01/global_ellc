"""
A generalised multi-instrument version of James Blake's emcee_combined.py

See the repository README.md for instructions

Contributors:
    James McCormac + James Blake
"""
import os
import sys
import argparse as ap
from datetime import datetime
from collections import defaultdict, OrderedDict
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
import numpy as np
import emcee
import corner
import ellc

# TODO: URGENT: double check the weights for combining phot and rvs
# TODO  URGENT: make ldcs filter specifc as expected
# TODO: eventually read in the Gaussian parameters
# TODO: eventually account for all other binary params (3rd light etc)

# use pylint as a syntax checker only
# pylint: disable=invalid-name
# pylint: disable=unused-variable
# pylint: disable=no-member
# pylint: disable=redefined-outer-name
# pylint: disable=superfluous-parens
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
# pylint: disable=line-too-long

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
    p.add_argument('config',
                   help='path to config file')
    p.add_argument('--threads',
                   help='number of threads to run',
                   type=int,
                   default=1)
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
            cols = [int(c) for c in sp[3:]]
            config[lc_file] = {'cols': cols}
            config['lcs'][filt].append(lc_file)
            continue
        elif sp[0] == 'rv':
            if "rvs" not in config.keys():
                config['rvs'] = OrderedDict()
            inst = sp[1]
            rv_file = sp[2]
            cols = [int(c) for c in sp[3:]]
            config[rv_file] = {'cols': cols}
            config['rvs'][inst] = rv_file
            continue
        elif sp[0] == 'nsteps':
            config['nsteps'] = int(sp[1])
            continue
        elif sp[0] == 'walker_scaling':
            config['walker_scaling'] = int(sp[1])
            continue
        elif sp[0] == 'thinning_factor':
            config['thinning_factor'] = int(sp[1])
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
    # make some checks for all parameters required
    # RVs + vsys
    if 'rvs' in config.keys() and 'vsys' in config['uniform_prior'].keys():
        try:
            assert len(config['rvs']) == len(config['uniform_prior']['vsys']), "Mismatching RV + Vsys!"
        except KeyError:
            print('Mismatch in RVs + matching Vsys values, exiting!')
            sys.exit(1)

    defaults = 0
    # nsteps for MCMC
    if 'nsteps' not in config.keys():
        print('nsteps not supplied in the config file, defaulting to 1000...')
        config['nsteps'] = 1000
        defaults += 1
    # walker_scaling
    if 'walker_scaling' not in config.keys():
        print('walker_scaling not supplied in the config file, defaulting to 1...')
        config['walker_scaling'] = 1
        defaults += 1
    # adds optional thinning factor for the MCMC sampling
    if 'thinning_factor' not in config.keys():
        print('thinning_factor for MCMC sampling not supplied, defaulting to 1 (no thinning)...')
        config['thinning_factor'] = 1
        defaults += 1
    if defaults > 0:
        x = raw_input('Accept the defaults above? (y | n): ')
        if x.lower() != 'y':
            print('Quiting!')
            sys.exit(1)
    if 'uniform_prior' not in config.keys():
        config['uniform_prior'] = []
    if 'fixed' not in config.keys():
        config['fixed'] = []
    if 'no_prior' not in config.keys():
        config['no_prior'] = []
    return config

def loadPhot(config):
    """
    Generic photometry loadng function

    This function assumes the file format is:
        time  measurment  error

    Parameters
    ----------
    config : array-like
        object containing all configuration parameters

    Returns
    -------
    x_data : array-like
        array of time data
    y_data : array-like
        array of measurement data
    yerr_data : array-like
        array of errors on measurments

    Raises
    ------
    None
    """
    x_data = OrderedDict()
    y_data = OrderedDict()
    yerr_data = OrderedDict()
    for filt in config['lcs']:
        x_dat, y_dat, yerr_dat = [], [], []
        for dat in config['lcs'][filt]:
            infile = "{}/{}".format(config['data_dir'], dat)
            x, y, e = np.loadtxt(infile, usecols=config[dat]['cols'], unpack=True)
            x_dat.append(x)
            y_dat.append(y)
            yerr_dat.append(e)
        # stack the light curves into the global holder
        x_data[filt] = np.hstack(x_dat)
        y_data[filt] = np.hstack(y_dat)
        yerr_data[filt] = np.hstack(yerr_dat)
    return x_data, y_data, yerr_data

def loadRvs(config):
    """
    Generic RV loadng function. This is subtley different
    to loadPhot as phot uses defaultdict and RVs use
    OrderedDict to hold the data.

    This function assumes the file format is:
        time  measurment  error

    Parameters
    ----------
    config : array-like
        object containing all configuration parameters

    Returns
    -------
    x_data : array-like
        array of time data
    y_data : array-like
        array of measurement data
    yerr_data : array-like
        array of errors on measurments

    Raises
    ------
    None
    """
    x_data = OrderedDict()
    y_data = OrderedDict()
    yerr_data = OrderedDict()
    for inst in config['rvs']:
        rv_file = config['rvs'][inst]
        infile = "{}/{}".format(config['data_dir'], rv_file)
        x, y, e = np.loadtxt(infile, usecols=config[rv_file]['cols'], unpack=True)
        x_data[inst] = x
        y_data[inst] = y
        yerr_data[inst] = e
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
        f_s = sqrt(e).sin(omega)
    f_c : float
        f_c = sqrt(e).cos(omega)
    a : float
        semi-major axis of binary in units of r1 (a/r1)
    q : float
        mass ratio of the binary (m2/m1)
    ldc_1 : array-like
        LDCs for primary eclipse ([ldc1_1, ldc1_2], assumes quadratic law)
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
        f_s = sqrt(e).sin(omega)
    f_c : float
        f_c = sqrt(e).cos(omega)
    a : float
        semi-major axis of binary in units of r1 (a/r1)
    q : float
        mass ratio of the binary (m2/m1)
    vsys : float
        systemtic velocity for the target

    Returns
    -------
    rv_model : array-like
        rv model of the binary using input parameters

    Raises
    ------
    None
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
                     grid_1='sparse',
                     grid_2='sparse',
                     f_c=f_c,
                     f_s=f_s,
                     flux_weighted=False)
    # account for the systemic velocity
    rv_model = rv1 + v_sys
    return rv_model

def lnprior(theta, config, n_priors):
    """
    Log prior function. This is used to ensure
    walkers are exploring the range defined by
    the priors. If not -np.inf is returned. Assuming
    all is fine a 0.0 is returned to allow the walkers
    to continue exploring the parameter space

    Parameters
    ----------
    theta : array-like
        current set of parameters from MCMC
    config : array-like
        object containing all configuration parameters
    n_priors : int
        number of parameters with priors

    Returns
    -------
    lnprior : float
        log prior of current sample (0.0 | -np.inf)

    Raises
    ------
    ValueError
        When prior limits are non-physical
    """
    priors = config['uniform_prior']
    for i, p in enumerate(priors):
        if p != 'vsys':
            val = theta[i]
            llim = priors[p]['prior_l']
            ulim = priors[p]['prior_h']
            # check for incorrect priors
            if llim > ulim:
                raise ValueError('{} priors wrong! {} > {}'.format(p, llim, ulim))
            if val < llim or val > ulim:
                return -np.inf
        imax = i
    # there are still some values to check, hence vsys values
    if imax < n_priors:
        # check RVs are included?
        if 'rvs' in config.keys():
            for j, p in enumerate(priors['vsys']):
                val = theta[imax+j]
                llim = priors['vsys'][p]['prior_l']
                ulim = priors['vsys'][p]['prior_h']
                # check for incorrect priors
                if llim > ulim:
                    raise ValueError('{} priors wrong! {} > {}'.format(p, llim, ulim))
                if val < llim or val > ulim:
                    return -np.inf
    return 0.0

def lnlike_sub(data_type, model, data, error):
    """
    Work out the log likelihood for a given subset of data

    Parameters
    ----------
    data_type : string
        type of data to evaluate (phot | rv)
    model : array-like
        current rv or phot model
    data : array-like
        data to compare to current model
    error : array-like
        error on the data measurements

    Returns
    -------
    lnlike : float
        log likelihood of model vs data

    Raises
    ------
    None
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
    This used lnlike_sub to do the data specific calculation

    Parameters
    ----------
    theta : array-like
        current set of parameters from MCMC
    config : array-like
        object containing all configuration parameters
    x_lc : array-like
        x element of photometry (time)
    y_lc : array-like
        y element of photometry (relative flux)
    yerr_lc : array-like
        yerr element of photometry (flux error)
    x_rv : array-like
        x element of rvs (time) | None
    y_rv : array-like
        y element of rvs (RV) | None
    yerr_rv : array-like
        yerr element of rvs (RV error) | None

    Returns
    -------
    lnlike : float
        combined lnlike for global model

    Raises
    ------
    IndexError
        Raised whenever a parameter cannot be found in theta/config
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
    if 'r2_r1' in no_prior or 'r2_r1' in uniform:
        r2_r1 = theta[params.index('r2_r1')]
    elif 'r2_r1' in fixed:
        r2_r1 = fixed['r2_r1']
    else:
        raise IndexError('Cannot find r2_r1 in lnlike')
    # sbratio
    if 'sbratio' in no_prior or 'sbratio' in uniform:
        sbratio = theta[params.index('sbratio')]
    elif 'sbratio' in fixed:
        sbratio = fixed['sbratio']
    else:
        raise IndexError('Cannot find sbratio in lnlike')
    # q
    if 'q' in no_prior or 'q' in uniform:
        q = theta[params.index('q')]
    elif 'q' in fixed:
        q = fixed['q']
    else:
        raise IndexError('Cannot find q in lnlike')
    # K -- assumed K2 for now
    if 'K' in no_prior or 'K' in uniform:
        K = theta[params.index('K')]
    elif 'K' in fixed:
        K = fixed['K']
    else:
        raise IndexError('Cannot find K in lnlike')
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
    # ldc1_1
    if 'ldc1_1' in no_prior or 'ldc1_1' in uniform:
        ldc1_1 = theta[params.index('ldc1_1')]
    elif 'ldc1_1' in fixed:
        ldc1_1 = fixed['ldc1_1']
    else:
        raise IndexError('Cannot find ldc1_1 in lnlike')
    # ldc1_2
    if 'ldc1_2' in no_prior or 'ldc1_2' in uniform:
        ldc1_2 = theta[params.index('ldc1_2')]
    elif 'ldc1_2' in fixed:
        ldc1_2 = fixed['ldc1_2']
    else:
        raise IndexError('Cannot find ldc1_2 in lnlike')
    # tweaking parameters
    ldcs_1 = [ldc1_1, ldc1_2]
    f_s = np.sqrt(ecc)*np.sin(omega*np.pi/180.)
    f_c = np.sqrt(ecc)*np.cos(omega*np.pi/180.)

    # derive some parameters from others
    # constants for 'a' and 'm' from Harmanec & Prsa, arXiv:1106.1508v2
    r2_a = r2_r1 * r1_a
    a_rs = (0.019771142*K*(1.+1./q)*period*np.sqrt(1.-ecc**2.)) / (np.sin(np.radians(incl)))
    b = np.cos(np.radians(incl)) / r1_a
    m_2 = 1.036149050206E-7*K*((K+K/q)**2)*period*((1.0-ecc**2.)**1.5) / (np.sin(np.radians(incl))**3.)
    m_1 = m_2/q
    #logg_1 = np.log(m_1) - (2.0*np.log(r1_a*a_rs)) + 4.437

    # Sanity check some parameters, inspiration taken from Liam's code
    if r2_r1 < 0:
        print('r2_r1 violation...')
        return -np.inf
    if r1_a < 0:
        print('r1_a violation...')
        return -np.inf
    if r2_a < 0:
        print('r2_a violation...')
        return -np.inf
    if b < 0 or b > 1+r2_r1 or b > 1/r1_a:
        print('b violation...')
        return -np.inf
    if ecc < 0 or ecc >= 1:
        print('ecc violation...')
        return -np.inf
    if K < 0:
        print('K violation...')
        return -np.inf
    if q < 0:
        print('q violation...')
        return -np.inf
    #if logg_1 < 0:
    #    print('logg_1 violation...')
    #    return -np.inf
    if m_2 < 0:
        print('m_2 violation...')
        return -np.inf
    if f_c < -1 or f_c > 1:
        print('f_c violation...')
        return -np.inf
    if f_s < -1 or f_s > 1:
        print('f_s violation...')
        return -np.inf
    # my priors
    if m_1 > 1.61 or m_1 < 1.11:
        print('m_1={:.3f}, m_2={:.5f} violation...'.format(m_1, m_2))
        return -np.inf

    # calculate lnlike of light curves
    lnlike_lc = 0.0
    for filt in x_lc:
        model_lc = light_curve_model(t_obs=x_lc[filt],
                                     t0=t0,
                                     period=period,
                                     radius_1=r1_a,
                                     radius_2=r2_a,
                                     sbratio=sbratio,
                                     a=a_rs,
                                     q=q,
                                     incl=incl,
                                     f_s=f_s,
                                     f_c=f_c,
                                     ldc_1=ldcs_1)
        lnlike_lc += lnlike_sub('phot', model_lc, y_lc[filt], yerr_lc[filt])

    # calculate lnlike of the radial velocities, if they exist
    if x_rv and y_rv and yerr_rv:
        lnlike_rv = 0.0
        for inst in x_rv:
            vsys = theta[params.index('vsys_{}'.format(inst))]
            model_rv = rv_curve_model(t_obs=x_rv[inst],
                                      t0=t0,
                                      period=period,
                                      radius_1=r1_a,
                                      radius_2=r2_a,
                                      sbratio=sbratio,
                                      a=a_rs,
                                      q=q,
                                      incl=incl,
                                      f_s=f_s,
                                      f_c=f_c,
                                      v_sys=vsys)
            lnlike_rv += lnlike_sub('rv', model_rv, y_rv[inst], yerr_rv[inst])

    # sum to get overall likelihood function, add RV stat if present
    if x_rv and y_rv and yerr_rv:
        lnlike = lnlike_lc + lnlike_rv
    else:
        lnlike = lnlike_lc
    return lnlike

def lnprob(theta, config, n_priors,
           x_lc, y_lc, yerr_lc,
           x_rv, y_rv, yerr_rv):
    """
    Log probability function. Wraps lnprior and lnlike

    Parameters
    ----------
    theta : array-like
        current set of parameters from MCMC
    config : array-like
        object containing all configuration parameters
    n_priors : int
        number of parameters with priors
    x_lc : array-like
        x element of photometry (time)
    y_lc : array-like
        y element of photometry (relative flux)
    yerr_lc : array-like
        yerr element of photometry (flux error)
    x_rv : array-like
        x element of rvs (time) | None
    y_rv : array-like
        y element of rvs (RV) | None
    yerr_rv : array-like
        yerr element of rvs (RV error) | None

    Returns
    -------
    lp : float
        log probability of the current model proposal

    Raises
    ------
    None
    """
    lp = lnprior(theta, config, n_priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, config,
                       x_lc, y_lc, yerr_lc,
                       x_rv, y_rv, yerr_rv)

def findBestParameter(param, config):
    """
    Locate the best (or fixed) value
    for a given parameter. Best fitting
    values and fixed parameters are all
    kept in config. Find them there.

    Parameters
    ----------
    param : string
        name of the parameter to look for
    config : array-like
        object containing all configuration parameters

    Returns
    -------
    param_value : float
        best fitting | fixed parameter value

    Raises
    ------
    IndexError
        whenever the parameter cannot be found in config
    """
    if param in best_params:
        return best_params[param]['value']
    elif param in config['fixed']:
        return config['fixed'][param]
    else:
        raise IndexError('Cannot find {} in best_parameters | fixed'.format(param))

if __name__ == "__main__":
    args = argParse()
    config = readConfig(args.config)
    outdir = config['out_dir']
    if not os.path.exists(outdir):
        os.mkdir(outdir)
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
    x_lc, y_lc, yerr_lc = loadPhot(config)
    if 'rvs' in config.keys():
        x_rv, y_rv, yerr_rv = loadRvs(config)
    else:
        x_rv, y_rv, yerr_rv = None, None, None
    # set up the sampler
    ndim = len(initial)
    # recommended nwalkers is 4*n_parameters
    # more walkers can help find the global minima, hence optional scaling
    nwalkers = 4*len(initial) * config['walker_scaling']
    # set the number of steps in the MCMC chain
    nsteps = config['nsteps']
    thinning_factor = config['thinning_factor']
    # set up the starting positions
    pos = [initial + weights*np.random.randn(ndim) for i in range(nwalkers)]

    # set up the sampler
    # if no RVs, pass None for each RV value so lnlike can take care of that
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(config, n_priors,
                                          x_lc, y_lc, yerr_lc,
                                          x_rv, y_rv, yerr_rv),
                                    threads=args.threads)
    # run the production chain
    tstart = datetime.utcnow()
    print("Running MCMC...")
    # run the sampler with the progress status
    #sampler.run_mcmc(pos, nsteps, rstate0=np.random.get_state())
    for i, result in enumerate(sampler.sample(pos, iterations=nsteps,
                                              rstate0=np.random.get_state(),
                                              thin=thinning_factor)):
        if (i+1) % 100 == 0:
            print("{0:5.1%}".format(float(i) / nsteps))
    print("Saving chain...")
    np.savetxt('{}/chain_{}steps_{}walkers.csv'.format(outdir, nsteps, nwalkers),
               np.c_[sampler.chain.reshape((-1, ndim))],
               delimiter=',', header=','.join(parameters))
    print("Done.")
    tend = datetime.utcnow()
    print('Time to complete: {}'.format(tend - tstart))
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
    # print them to screen and log them to disc
    best_params = OrderedDict()
    logfile = "{}/best_fitting_params.txt".format(outdir)
    best_pars_index = np.unravel_index(np.argmax(sampler.lnprobability),
                                       (nwalkers, nsteps/thinning_factor))
    best_pars = sampler.chain[best_pars_index[0], best_pars_index[1], :]
    with open(logfile, 'w') as lf:
        for i, param in enumerate(config['parameters']):
            best_params[param] = {'value': best_pars[i],
                                  'error': np.std(samples[:, i])}
            line = "{}: {:.6f} +/- {:.6f}".format(param,
                                                  best_params[param]['value'],
                                                  best_params[param]['error'])
            print(line)
            lf.write(line+"\n")
    # stick the best params in the config with everything else
    config['best_params'] = best_params
    # make a corner plot
    labels = ["$"+p+"$" for p in config['parameters']]
    fig = corner.corner(samples,
                        labels=labels,
                        truths=initial,
                        plot_contours=False)
    fig.savefig('{}/corner_{}steps_{}walkers.png'.format(outdir, nsteps, nwalkers))
    fig.clf()

    # extract the final parameters in a generic way
    # to plot the final model and data together
    sbratio = findBestParameter('sbratio', config)
    r1_a = findBestParameter('r1_a', config)
    r2_r1 = findBestParameter('r2_r1', config)
    incl = findBestParameter('incl', config)
    t0 = findBestParameter('t0', config)
    period = findBestParameter('period', config)
    ecc = findBestParameter('ecc', config)
    omega = findBestParameter('omega', config)
    ldc1_1 = findBestParameter('ldc1_1', config)
    ldc1_2 = findBestParameter('ldc1_2', config)
    q = findBestParameter('q', config)
    K = findBestParameter('K', config)

    # take most likely set of parameters and plot the models
    # make a dense mesh of time points for the lcs and RVs
    # this is done in phase space for simplicity,
    # i.e. P = 1 and T0 = 0.0 in model

    # set up some param combos for plotting
    f_s = np.sqrt(ecc)*np.sin(omega*np.pi/180.)
    f_c = np.sqrt(ecc)*np.cos(omega*np.pi/180.)
    ldcs_1 = [ldc1_1, ldc1_2]

    # derive some parameters from others
    r2_a = r2_r1 * r1_a
    a_rs = (0.019771142*K*(1.+1./q)*period*np.sqrt(1.-ecc**2.)) / (np.sin(np.radians(incl)))
    print('a_rsun: {:.4}'.format(a_rs))

    # set up the plot
    num_plots = len(config['lcs']) + 1
    fig, ax = plt.subplots(num_plots, 1, figsize=(15, 15))
    colours = ['k.', 'r.', 'g.', 'b.', 'c.']
    colours_rvs = ['ko', 'ro', 'go', 'bo', 'co']
    pn = 0

    # assumed we always have at least photometry
    # final models
    x_model = np.linspace(-0.5, 0.5, 1000)
    for filt in config['lcs']:
        final_lc_model = light_curve_model(t_obs=x_model,
                                           t0=0.0,
                                           period=1.0,
                                           radius_1=r1_a,
                                           radius_2=r2_a,
                                           sbratio=sbratio,
                                           a=a_rs,
                                           q=q,
                                           incl=incl,
                                           f_s=f_s,
                                           f_c=f_c,
                                           ldc_1=ldcs_1)
        phase_lc = ((x_lc[filt] - t0)/period)%1
        ax[pn].plot(phase_lc, y_lc[filt], 'k.')
        ax[pn].plot(phase_lc-1, y_lc[filt], 'k.')
        ax[pn].plot(x_model, final_lc_model, 'g-', lw=2)
        ax[pn].set_xlim(-0.04, 0.04)
        ax[pn].set_ylim(0.99, 1.01)
        ax[pn].set_xlabel('Orbital Phase')
        ax[pn].set_ylabel('Relative Flux')
        pn += 1

    # plot RVs if we have them
    if 'rvs' in config.keys():
        # pick a reference instrument for scaling RVs
        # to match the systemtic velocities
        ref_inst = config['rvs'].keys()[0]
        vsys_ref = best_params['vsys_{}'.format(ref_inst)]['value']
        x_rv_model = np.linspace(t0, t0+period, 1000)
        phase_rv_model = ((x_rv_model-t0)/period)%1
        final_rv_model = rv_curve_model(t_obs=x_rv_model,
                                        t0=t0,
                                        period=period,
                                        radius_1=r1_a,
                                        radius_2=r2_a,
                                        sbratio=sbratio,
                                        a=a_rs,
                                        q=q,
                                        incl=incl,
                                        f_s=f_s,
                                        f_c=f_c,
                                        v_sys=vsys_ref)
        # plot the RVs + model
        for i, inst in enumerate(config['rvs']):
            phase_rv = ((x_rv[inst] - t0)/period)%1
            if inst == ref_inst:
                ax[pn].errorbar(phase_rv, y_rv[inst], yerr=yerr_rv[inst], fmt=colours[i])
            else:
                vsys_diff = vsys_ref - best_params['vsys_{}'.format(inst)]['value']
                ax[pn].errorbar(phase_rv, y_rv[inst] + vsys_diff,
                                yerr=yerr_rv[inst], fmt=colours[i])
        ax[pn].plot(phase_rv_model, final_rv_model, 'r-', lw=2)
        ax[pn].set_xlim(0, 1)
        ax[pn].set_xlabel('Orbital Phase')
        ax[pn].set_ylabel('Radial Velocity')

    # save the final model fit
    fig.savefig('{}/chain_{}steps_{}walkers_fitted_models.png'.format(outdir,
                                                                      nsteps,
                                                                      nwalkers))
