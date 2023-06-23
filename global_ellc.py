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
import matplotlib.pyplot as plt
import numpy as np
import emcee
import corner
import ellc

# TODO: add outputing of m_2 and a_rs to file with errors

# use pylint as a syntax checker only
# pylint: disable=invalid-name
# pylint: disable=unused-variable
# pylint: disable=no-member
# pylint: disable=redefined-outer-name
# pylint: disable=superfluous-parens
# pylint: disable=too-many-arguments pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
# pylint: disable=line-too-long

# Constants - taken from https://arxiv.org/pdf/1510.07674.pdf
MSUN = 1.3271244E20/6.67408E-11 # kg
RSUN = 6.957E8 # m
MJUP = 1.2668653E17/6.67408E-11 # kg
RJUP = 7.1492E7 # m
AU = 1.496E11 # m

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
    p.add_argument('--v',
                   help='increase verbosity',
                   action='store_true')
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
    # set up empty lists
    config['parameters'] = []
    config['initials'] = []
    config['weights'] = []
    # count the number of priors
    config['n_priors'] = 0
    # read in the data
    f = open(infile).readlines()
    for line in f:
        print(line.rstrip())
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
        elif sp[0] == 'best_selection':
            if sp[1] in ['median', 'argmax']:
                config['best_selection'] = sp[1]
                continue
        elif sp[0] == 'r2_jup_limit':
            # used for grazing systems to kill unphysical planet solutions
            config['r2_jup_limit'] = float(sp[1])
            continue
        # read Fixed parameters
        if sp[1] == 'F':
            if 'fixed' not in config.keys():
                config['fixed'] = OrderedDict()
            if sp[0] == 'vsys' or sp[0] == 'ldc1_1' or sp[0] == 'ldc1_2' or sp[0] == 'light_3':
                try:
                    param = sp[0]
                    inst_filt = sp[2]
                    param += "_{}".format(inst_filt)
                    value = float(sp[3])
                except IndexError:
                    print("Fixed prior request not formatted correctly")
                    print('PARAM F INST_FILT VALUE')
                    print(line)
                    sys.exit(1)
                config['fixed'][param] = {'value': value}
            else:
                try:
                    param = sp[0]
                    value = float(sp[2])
                except IndexError:
                    print("Fixed prior request not formatted correctly")
                    print('PARAM F VALUE')
                    print(line)
                    sys.exit(1)
                config['fixed'][param] = {'value': value}
        # read fit parameters with No prior
        elif sp[1] == 'N':
            if 'no_prior' not in config.keys():
                config['no_prior'] = OrderedDict()
            if sp[0] == 'vsys' or sp[0] == 'ldc1_1' or sp[0] == 'ldc1_2' or sp[0] == 'light_3':
                try:
                    param = sp[0]
                    inst_filt = sp[2]
                    param += "_{}".format(inst_filt)
                    value = float(sp[3])
                    weight = float(sp[4])
                except IndexError:
                    print("No prior request not formatted correctly")
                    print('PARAM N INST_FILT VALUE WEIGHT')
                    print(line)
                    sys.exit(1)
                config['no_prior'][param] = {'value': value,
                                             'weight': weight}
            else:
                try:
                    param = sp[0]
                    value = float(sp[2])
                    weight = float(sp[3])
                except IndexError:
                    print("No prior request not formatted correctly")
                    print('PARAM N VALUE WEIGHT')
                    print(line)
                    sys.exit(1)
                config['no_prior'][param] = {'value': value,
                                             'weight': weight}
            # append to the list to step over
            config['parameters'].append(param)
            config['initials'].append(value)
            config['weights'].append(weight)
        # read fit parameters with Uniform priors
        elif sp[1] == 'U':
            if 'uniform_prior' not in config.keys():
                config['uniform_prior'] = OrderedDict()
            if sp[0] == 'vsys' or sp[0] == 'ldc1_1' or sp[0] == 'ldc1_2' or sp[0] == 'light_3':
                try:
                    param = sp[0]
                    inst_filt = sp[2]
                    param += "_{}".format(inst_filt)
                    value = float(sp[3])
                    weight = float(sp[4])
                    prior_l = float(sp[5])
                    prior_h = float(sp[6])
                except IndexError:
                    print("Uniform prior request not formatted correctly")
                    print('PARAM U INST_FILT VALUE WEIGHT PRIOR_LOW PRIOR_HIGH')
                    print(line)
                    sys.exit(1)
                config['uniform_prior'][param] = {'value': value,
                                                  'weight': weight,
                                                  'prior_l': prior_l,
                                                  'prior_h': prior_h}
            else:
                try:
                    param = sp[0]
                    value = float(sp[2])
                    weight = float(sp[3])
                    prior_l = float(sp[4])
                    prior_h = float(sp[5])
                except IndexError:
                    print("Uniform prior request not formatted correctly")
                    print('PARAM U VALUE WEIGHT PRIOR_LOW PRIOR_HIGH')
                    print(line)
                    sys.exit(1)
                if 'uniform_prior' not in config.keys():
                    config['uniform_prior'] = OrderedDict()
                config['uniform_prior'][param] = {'value': value,
                                                  'weight': weight,
                                                  'prior_l': prior_l,
                                                  'prior_h': prior_h}
            # append to the list to step over
            config['parameters'].append(param)
            config['initials'].append(value)
            config['weights'].append(weight)
            config['n_priors'] += 1
        elif sp[1] == 'E':
            if 'external_prior' not in config.keys():
                config['external_prior'] = OrderedDict()
            try:
                param = sp[0]
                value = float(sp[2])
                weight = float(sp[3])
            except IndexError:
                print("External prior request not formatted correctly")
                print('PARAM E VALUE WEIGHT')
                print(line)
                sys.exit(1)
            config['external_prior'][param] = {'value': value,
                                               'weight': weight}

    # checks for some sensible defaults
    # nsteps for MCMC
    if 'nsteps' not in config.keys():
        print('nsteps not supplied in the config file, defaulting to 1000...')
        config['nsteps'] = 1000
    # walker_scaling
    if 'walker_scaling' not in config.keys():
        print('walker_scaling not supplied in the config file, defaulting to 1...')
        config['walker_scaling'] = 1
    # adds optional thinning factor for the MCMC sampling
    if 'thinning_factor' not in config.keys():
        print('thinning_factor for MCMC sampling not supplied, defaulting to 1 (no thinning)...')
        config['thinning_factor'] = 1
    # adds missing best selection, defaults to median
    if 'best_selection' not in config.keys():
        print('best parameter selection method for MCMC is missing, defaulting to median...')
        config['best_selection'] = 'median'

    # adds missing keys if not used - so they can be checked later and not break the code
    if 'uniform_prior' not in config.keys():
        config['uniform_prior'] = []
    if 'fixed' not in config.keys():
        config['fixed'] = []
    if 'no_prior' not in config.keys():
        config['no_prior'] = []
    if 'external_prior' not in config.keys():
        config['external_prior'] = []
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
                      spots_1=None, spots_2=None, light_3=0.0):
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
        semi-major axis of binary in units of rsun
    q : float
        mass ratio of the binary (m2/m1)
    ldc_1 : array-like
        LDCs for primary eclipse ([ldc1_1, ldc1_2], assumes quadratic law)
    spots_1 : array-like
        spot parameters for spot_1 [check order!]
    spots_2 : array-like
        spot parameters for spot_2 [check order!]
    light_3 : float
        3rd light component for this band
        default = 0.0

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
                       spots_2=spots_2,
                       light_3=light_3)
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
        semi-major axis of binary in units of rsun
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

def lnprior(theta, config):
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
    params = config['parameters']
    for p in priors:
        val = theta[params.index(p)]
        llim = priors[p]['prior_l']
        ulim = priors[p]['prior_h']
        # check for incorrect priors
        if llim > ulim:
            raise ValueError('{} priors wrong! {} > {}'.format(p, llim, ulim))
        if val < llim or val > ulim:
            #print(p, llim, val, ulim)
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
    external = config['external_prior']

    # t0
    if 't0' in no_prior or 't0' in uniform:
        t0 = theta[params.index('t0')]
    elif 't0' in fixed:
        t0 = fixed['t0']['value']
    else:
        raise IndexError('Cannot find t0 in lnlike')

    # period
    if 'period' in no_prior or 'period' in uniform:
        period = theta[params.index('period')]
    elif 'period' in fixed:
        period = fixed['period']['value']
    else:
        raise IndexError('Cannot find period in lnlike')

    # radius_1
    if 'r1_a' in no_prior or 'r1_a' in uniform:
        r1_a = theta[params.index('r1_a')]
    elif 'r1_a' in fixed:
        r1_a = fixed['r1_a']['value']
    else:
        raise IndexError('Cannot find r1_a in lnlike')

    # radius_2
    if 'r2_r1' in no_prior or 'r2_r1' in uniform:
        r2_r1 = theta[params.index('r2_r1')]
    elif 'r2_r1' in fixed:
        r2_r1 = fixed['r2_r1']['value']
    else:
        raise IndexError('Cannot find r2_r1 in lnlike')

    # sbratio
    if 'sbratio' in no_prior or 'sbratio' in uniform:
        sbratio = theta[params.index('sbratio')]
    elif 'sbratio' in fixed:
        sbratio = fixed['sbratio']['value']
    else:
        raise IndexError('Cannot find sbratio in lnlike')

    # q
    if 'q' in no_prior or 'q' in uniform:
        q = theta[params.index('q')]
    elif 'q' in fixed:
        q = fixed['q']['value']
    else:
        raise IndexError('Cannot find q in lnlike')

    # K -- assumed K2 for now
    if 'K' in no_prior or 'K' in uniform:
        K = theta[params.index('K')]
    elif 'K' in fixed:
        K = fixed['K']['value']
    else:
        raise IndexError('Cannot find K in lnlike')

    # incl
    if 'incl' in no_prior or 'incl' in uniform:
        incl = theta[params.index('incl')]
    elif 'incl' in fixed:
        incl = fixed['incl']['value']
    else:
        raise IndexError('Cannot find incl in lnlike')

    # ecc
    if 'ecc' in no_prior or 'ecc' in uniform:
        ecc = theta[params.index('ecc')]
    elif 'ecc' in fixed:
        ecc = fixed['ecc']['value']
    else:
        raise IndexError('Cannot find ecc in lnlike')

    # omega
    if 'omega' in no_prior or 'omega' in uniform:
        omega = theta[params.index('omega')]
    elif 'omega' in fixed:
        omega = fixed['omega']['value']
    else:
        raise IndexError('Cannot find omega in lnlike')

    # set up some combined params
    f_s = np.sqrt(ecc)*np.sin(omega*np.pi/180.)
    f_c = np.sqrt(ecc)*np.cos(omega*np.pi/180.)

    # derive some parameters from others
    # constants for 'a' and 'm' from Harmanec & Prsa, arXiv:1106.1508v2
    r2_a = r2_r1 * r1_a

    # semi-major axis in solar radii
    a_rs = (0.019771142*K*(1.+1./q)*period*np.sqrt(1.-ecc**2.)) / (np.sin(np.radians(incl)))
    b = np.cos(np.radians(incl)) / r1_a
    # secondary mass in solar masses
    m_2 = 1.036149050206E-7*K*((K+K/q)**2)*period*((1.0-ecc**2.)**1.5) / (np.sin(np.radians(incl))**3.)
    # primary mass in solar masses
    m_1 = m_2/q
    logg_1 = np.log(m_1) - (2.0*np.log(r1_a*a_rs)) + 4.437

    # work out r2 in m - this can be used to apply a cut on
    # r2 > physical planet size, e.g. > 3Rjup
    r_2_m = (r2_a * a_rs * RSUN)
    r_2_jup = r_2_m / RJUP

    # apply a max cut on the secondary in r_jup - can be used if grazing planet
    if 'r2_jup_limit' in config.keys():
        if r_2_jup > config['r2_jup_limit']:
            if args.v:
                print('r2_jup > limit {} - violation...'.format(config['r2_jup_limit']))
            return -np.inf

    # Sanity check some parameters, inspiration taken from Liam's code
    if r2_r1 < 0:
        if args.v:
            print('r2_r1 violation...')
        return -np.inf
    if r1_a < 0:
        if args.v:
            print('r1_a violation...')
        return -np.inf
    if r2_a < 0:
        if args.v:
            print('r2_a violation...')
        return -np.inf
    if b < 0 or b > 1+r2_r1 or b > 1/r1_a:
        if args.v:
            print('b violation...')
        return -np.inf
    if ecc < 0 or ecc >= 1:
        if args.v:
            print('ecc violation...')
        return -np.inf
    if K < 0:
        if args.v:
            print('K violation...')
        return -np.inf
    if q < 0:
        if args.v:
            print('q violation...')
        return -np.inf
    if logg_1 < 0:
        if args.v:
            print('logg_1 violation...')
        return -np.inf
    if m_2 < 0:
        if args.v:
            print('m_2 violation...')
        return -np.inf
    if f_c < -1 or f_c > 1:
        if args.v:
            print('f_c violation...')
        return -np.inf
    if f_s < -1 or f_s > 1:
        if args.v:
            print('f_s violation...')
        return -np.inf

    # calculate lnlike of light curves
    lnlike_lc = 0.0
    for filt in x_lc:
        ldc1_1_keyword = "ldc1_1_{}".format(filt)
        ldc1_2_keyword = "ldc1_2_{}".format(filt)
        light_3_keyword = "light_3_{}".format(filt)
        # ldc1_1_filt
        if ldc1_1_keyword in no_prior or ldc1_1_keyword in uniform:
            ldc1_1 = theta[params.index(ldc1_1_keyword)]
        elif ldc1_1_keyword in fixed:
            ldc1_1 = fixed[ldc1_1_keyword]['value']
        else:
            raise IndexError('Cannot find {} in lnlike'.format(ldc1_1_keyword))
        # ldc1_2_filt
        if ldc1_2_keyword in no_prior or ldc1_2_keyword in uniform:
            ldc1_2 = theta[params.index(ldc1_2_keyword)]
        elif ldc1_2_keyword in fixed:
            ldc1_2 = fixed[ldc1_2_keyword]['value']
        else:
            raise IndexError('Cannot find {} in lnlike'.format(ldc1_2_keyword))
        # light_3_filt
        if light_3_keyword in no_prior or light_3_keyword in uniform:
            light_3 = theta[params.index(light_3_keyword)]
        elif light_3_keyword in fixed:
            light_3 = fixed[light_3_keyword]['value']
        else:
            raise IndexError('Cannot find {} in lnlike'.format(light_3_keyword))
        # tweaking parameters
        ldcs_1 = [ldc1_1, ldc1_2]
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
                                     ldc_1=ldcs_1,
                                     light_3=light_3)
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

    # External priors: inspiration from Liam's code to use Gaussians
    # priors from spectroscopy to constrain the fit
    lnpriors_external = 0

    # check if we have any external priors
    if external:
        # stellar mass prior
        if 'm1' in external.keys():
            m_1_pr_0 = external['m1']['value']
            m_1_pr_s = external['m1']['weight']
            ln_m_1 = -0.5*(((m_1-m_1_pr_0)/m_1_pr_s)**2 + np.log(m_1_pr_s**2))
            lnpriors_external += ln_m_1

        # stellar radius prior
        if 'r1' in external.keys():
            r_1_pr_0 = external['r1']['value']
            r_1_pr_s = external['r1']['weight']
            ln_r_1 = -0.5*(((r1_a*a_rs-r_1_pr_0)/r_1_pr_s)**2 + np.log(r_1_pr_s**2))
            lnpriors_external += ln_r_1

        # stellar logg prior
        if 'logg1' in external.keys():
            logg_1_pr_0 = external['logg1']['value']
            logg_1_pr_s = external['logg1']['weight']
            ln_logg_1 = -0.5*(((logg_1-logg_1_pr_0)/logg_1_pr_s)**2 + np.log(logg_1_pr_s**2))
            lnpriors_external += ln_logg_1

        # planet density priorm used for grazing systems
        if 'den2' in external.keys():
            den_2_pr_0 = external['den2']['value']
            den_2_pr_s = external['den2']['weight']
            # get m_2 in grams for the density prior
            m_2_grams = m_2 * MSUN * 1000.
            r_2_cm = r2_a * a_rs * RSUN * 100
            den_2 = (3.*m_2_grams) / (4.*np.pi*(r_2_cm**3.))
            ln_den_2 = -0.5*(((den_2-den_2_pr_0)/den_2_pr_s)**2 + np.log(den_2_pr_s**2))
            lnpriors_external += ln_den_2

    # create the final lnlike
    lnlike = 0
    # add the likelihood from the photometry
    lnlike += lnlike_lc
    # add RV lnlike if present
    if x_rv and y_rv and yerr_rv:
        lnlike += lnlike_rv
    # add the lnlike from any external priors
    lnlike += lnpriors_external

    return lnlike

def lnprob(theta, config,
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
    lp = lnprior(theta, config)
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
        return config['fixed'][param]['value']
    else:
        raise IndexError('Cannot find {} in best_parameters | fixed'.format(param))

if __name__ == "__main__":
    args = argParse()
    config = readConfig(args.config)
    outdir = config['out_dir']
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # READ IN THE DATA
    x_lc, y_lc, yerr_lc = loadPhot(config)
    if 'rvs' in config.keys():
        x_rv, y_rv, yerr_rv = loadRvs(config)
    else:
        x_rv, y_rv, yerr_rv = None, None, None
    # set up the sampler
    ndim = len(config['initials'])
    # recommended nwalkers is 4*n_parameters
    # more walkers can help find the global minima, hence optional scaling
    nwalkers = 4*ndim*config['walker_scaling']
    # set the number of steps in the MCMC chain
    nsteps = config['nsteps']
    thinning_factor = config['thinning_factor']
    # set up the starting positions
    pos = [config['initials'] + config['weights']*np.random.randn(ndim) for i in range(nwalkers)]

    # set up the sampler
    # if no RVs, pass None for each RV value so lnlike can take care of that
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(config,
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
               delimiter=',', header=','.join(config['parameters']))
    print("Done.")
    tend = datetime.utcnow()
    print('Time to complete: {}'.format(tend - tstart))
    # plot and save the times series of each parameter
    for i, (initial_param, label) in enumerate(zip(config['initials'],
                                                   config['parameters'])):
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
    burnin = int(input('Enter burnin period: '))
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    # determine the most likely set of parameters
    # print them to screen and log them to disc
    best_params = OrderedDict()
    logfile = "{}/best_fitting_params_{}nsteps_{}nwalkers.txt".format(outdir, nsteps, nwalkers)
    with open(logfile, 'w') as lf:
        # work out the argmax index
        best_pars_index = np.unravel_index(np.argmax(sampler.lnprobability),
                                           (nwalkers, int(nsteps/thinning_factor)))
        best_pars = sampler.chain[best_pars_index[0], best_pars_index[1], :]
        # loop over the parameters - pick the best ones using the selected method
        for i, param in enumerate(config['parameters']):
            if config['best_selection'] == 'argmax':
                best_params[param] = {'value': best_pars[i],
                                      'error': np.std(samples[:, i])}
            else:
                best_params[param] = {'value': np.median(samples[:, i]),
                                      'error': np.std(samples[:, i])}
            line = "{}: {:.6f} +/- {:.6f}".format(param,
                                                  best_params[param]['value'],
                                                  best_params[param]['error'])
            print(line)
            lf.write(line+"\n")

    # stick the best params in the config with everything else
    config['best_params'] = best_params
    # make a corner plot
    labels = ["$"+p.replace('_','')+"$" for p in config['parameters']]
    fig = corner.corner(samples,
                        labels=labels,
                        truths=config['initials'],
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
    q = findBestParameter('q', config)
    K = findBestParameter('K', config)

    # take most likely set of parameters and plot the models
    # make a dense mesh of time points for the lcs and RVs
    # this is done in phase space for simplicity,
    # i.e. P = 1 and T0 = 0.0 in model

    # set up some param combos for plotting
    f_s = np.sqrt(ecc)*np.sin(omega*np.pi/180.)
    f_c = np.sqrt(ecc)*np.cos(omega*np.pi/180.)

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
        ldc1_1 = findBestParameter('ldc1_1_{}'.format(filt), config)
        ldc1_2 = findBestParameter('ldc1_2_{}'.format(filt), config)
        light_3 = findBestParameter('light_3_{}'.format(filt), config)
        ldcs_1 = [ldc1_1, ldc1_2]
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
                                           ldc_1=ldcs_1,
                                           light_3=light_3)
        phase_lc = ((x_lc[filt] - t0)/period)%1
        ax[pn].plot(phase_lc, y_lc[filt], 'k.')
        ax[pn].plot(phase_lc-1, y_lc[filt], 'k.')
        ax[pn].plot(x_model, final_lc_model, 'g-', lw=2)
        ax[pn].set_xlim(-0.15, 0.15)
        # work out the y limit
        yllim = np.median(sorted(y_lc[filt])[:21]) - 0.01
        yulim = np.median(sorted(y_lc[filt])[-21:]) + 0.01
        ax[pn].set_ylim(yllim, yulim)
        ax[pn].set_xlabel('Orbital Phase')
        ax[pn].set_ylabel('Relative Flux')
        ax[pn].set_title(filt)
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

