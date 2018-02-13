"""
Code to plot the RVs with model and residuals for papers
"""
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
from matplotlib import rc, cycler
import numpy as np
import ellc

# pylint: disable=invalid-name
# pylint: disable=redefined-outer-name

plt.style.use(['seaborn-white', 'seaborn-paper'])

# DATA GLOBALS
DATA_LINE_WIDTH = 1.0

# AXES GLOBALS
AXES_LINE_WIDTH = 1.0
AXES_MAJOR_TICK_LENGTH = 10
AXES_MINOR_TICK_LENGTH = AXES_MAJOR_TICK_LENGTH/2.
AXES_TICK_DIRECTION = 'in'

# FIGURE GLOBALS
ONE_COL_WIDTH = 3.46
TWO_COL_WIDTH = 7.09
DPI = 800

def general():
    """
    General settings for all plot types. Call this first,
    then call the cascading style required
    """
    rc('font', family='Times New Roman', size=10)
    rc('text', color='black')
    rc('figure', dpi=DPI)
    rc('lines', linestyle='-',
       linewidth=DATA_LINE_WIDTH)
    rc('axes',
       linewidth=AXES_LINE_WIDTH,
       titlesize=10,
       labelsize=10,
       prop_cycle=cycler('color', ['black']))
    rc('axes.formatter', limits=(-4, 4))
    rc('xtick',
       direction=AXES_TICK_DIRECTION,
       labelsize=10)
    rc('xtick.major',
       size=AXES_MAJOR_TICK_LENGTH,
       width=AXES_LINE_WIDTH)
    rc('xtick.minor',
       visible=True,
       size=AXES_MINOR_TICK_LENGTH,
       width=AXES_LINE_WIDTH)
    rc('ytick',
       direction=AXES_TICK_DIRECTION,
       labelsize=10)
    rc('ytick.major',
       size=AXES_MAJOR_TICK_LENGTH,
       width=AXES_LINE_WIDTH)
    rc('ytick.minor',
       visible=True,
       size=AXES_MINOR_TICK_LENGTH,
       width=AXES_LINE_WIDTH)

def one_column():
    """
    One-column-width plot settings
    """
    rc('figure', figsize=(ONE_COL_WIDTH, ONE_COL_WIDTH))

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

if __name__ == "__main__":
    # global parameters
    sbratio = 0.0           # fixed = set in lnlike
    radius_1 = 0.029363    #solar radii
    radius_2 = 0.004665     #solar radii
    incl = 89.6232
    t0 = 2453592.74192
    period = 16.9535452
    ecc = 0.16035
    omega = 78.39513
    a = 31.650747           #solar radii
    ldc_1_1 = 0.3897
    ldc_1_2 = 0.1477
    v_sys1 = -21.133
    v_sys2 = -21.122
    v_sys2_diff = -0.01098
    v_sys3 = -20.896
    v_sys3_diff = -0.23688
    q = 0.09649

    # orbital params
    f_s = np.sqrt(ecc)*np.sin(omega*np.pi/180.)
    f_c = np.sqrt(ecc)*np.cos(omega*np.pi/180.)

    # data and nights to plot
    datadir = "/Users/jmcc/Dropbox/EBLMs/J23431841"
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
    # get the final RV model
    x_rv_model = np.linspace(t0, t0+period, 1000)
    phase_rv_model = ((x_rv_model-t0)/period)%1
    final_rv_model = rv_curve_model(t_obs=x_rv_model,
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
    # phase the times
    phase_rv1 = ((x_rv1 - t0)/period)%1
    phase_rv2 = ((x_rv2 - t0)/period)%1
    phase_rv3 = ((x_rv3 - t0)/period)%1
    phase_rv_model = ((x_rv_model-t0)/period)%1

    # fies residuals
    res_rv_model_1 = rv_curve_model(t_obs=x_rv1,
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
    res_1 = y_rv1 - res_rv_model_1
    # sophie residuala
    res_rv_model_2 = rv_curve_model(t_obs=x_rv2,
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
    res_2 = y_rv2 - res_rv_model_2
    # paras residuals
    res_rv_model_3 = rv_curve_model(t_obs=x_rv3,
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
    res_3 = y_rv3 - res_rv_model_3



    # start plotting
    one_column()
    general()
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].errorbar(phase_rv1, y_rv1, yerr=yerr_rv1, fmt='k.')
    ax[0].errorbar(phase_rv2, y_rv2 + v_sys2_diff, yerr=yerr_rv2, fmt='g.')
    ax[0].errorbar(phase_rv3, y_rv3 + v_sys3_diff, yerr=yerr_rv3, fmt='b.')
    ax[0].plot(phase_rv_model[:-1], final_rv_model[:-1], 'r-', lw=1)
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(-31.0, -10.0)
    ax[0].set_ylabel('Radial Velocity (km/s)')
    ax[0].legend(('Model', 'FIES', 'SOPHIE', 'PARAS'),)
    ax[1].errorbar(phase_rv1, res_1, yerr=yerr_rv1, fmt='k.')
    ax[1].errorbar(phase_rv2, res_2, yerr=yerr_rv2, fmt='g.')
    ax[1].errorbar(phase_rv3, res_3, yerr=yerr_rv3, fmt='b.')
    ax[1].set_ylabel('Radial Velocity (km/s)')
    ax[1].set_xlabel('Orbital Phase')
    ax[1].set_ylim(-0.15, 0.15)
    ax[1].axhline(0, lw=1, c='r')
    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.20, right=0.94, hspace=0.0)
    fig.savefig('{}/all_rvs_with_model.png'.format(datadir), dpi=300)
