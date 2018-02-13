"""
Code to take some known model params
and plot individual nites transits and
residuals with the model
"""
from collections import defaultdict
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

def split_into_nights(t, m, e):
    """
    split up the combined lc file into
    the different nights for plotting
    """
    tn = defaultdict(list)
    mn = defaultdict(list)
    en = defaultdict(list)
    for i, j, k in zip(t, m, e):
        tn[int(i)].append(i)
        mn[int(i)].append(j)
        en[int(i)].append(k)
    return tn, mn, en

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

    # data and nights to plot
    data_dir = "/Users/jmcc/Dropbox/EBLMs/J23431841"
    data_files = ['NITES_J234318.41_Clear_20120829_F1_A14_all.lc.txt',
                  'NITES_J234318.41_Clear_20120829_F1_A14.lc.txt',
                  'NITES_J234318.41_Clear_20130923_F2_A14.lc.txt',
                  'NITES_J234318.41_Clear_20131010_F1_A14.lc.txt',
                  'NITES_J234318.41_Clear_20141001_F1_A14.lc.txt']

    one_column()
    general()
    fig, ax = plt.subplots(1)
    ax.set_xlim(-0.02, 0.02)
    ax.set_xlabel('Orbital phase')
    ax.set_ylabel('Flux ratio')

    offset = 0.05
    counter = 0
    # loop over the lcs and plot them
    for i, dat in enumerate(data_files):
        infile = "{}/{}".format(data_dir, dat)

        t, f, e = np.loadtxt(infile, usecols=[2, 3, 4], unpack=True)

        # model - this uses the params above and is copied from
        # the modelling script emcee_combined_ellc.py
        x_model = np.linspace(-0.2, 0.2, 1000)
        f_s = np.sqrt(ecc)*np.sin(omega*np.pi/180.)
        f_c = np.sqrt(ecc)*np.cos(omega*np.pi/180.)
        ldcs_1 = [ldc_1_1, ldc_1_2]
        final_lc_model = light_curve_model(t_obs=x_model,
                                           t0=0.0,
                                           period=1.0,
                                           radius_1=radius_1,
                                           radius_2=radius_2,
                                           sbratio=sbratio,
                                           a=a,
                                           q=q,
                                           incl=incl,
                                           f_s=f_s,
                                           f_c=f_c,
                                           ldc_1=ldcs_1)


        ph = ((t - t0)/period)%1
        lc_model_for_res = light_curve_model(t_obs=ph,
                                             t0=0.0,
                                             period=1.0,
                                             radius_1=radius_1,
                                             radius_2=radius_2,
                                             sbratio=sbratio,
                                             a=a,
                                             q=q,
                                             incl=incl,
                                             f_s=f_s,
                                             f_c=f_c,
                                             ldc_1=ldcs_1)

        res = f - lc_model_for_res

        # plot the data, models and residuals
        residual_offset = 0.80
        label_offset = 0.003
        if i > 0:
            ax.scatter(ph, f-(counter*offset), c='k', marker='.', s=1)
            ax.scatter(ph-1, f-(counter*offset), c='k', marker='.', s=1)
            ax.scatter(ph, res+residual_offset-(counter*offset/1.5), c='k', marker='.', s=1)
            ax.scatter(ph-1, res+residual_offset-(counter*offset/1.5), c='k', marker='.', s=1)
            ax.axhline(residual_offset-(counter*offset/1.5), c='r', lw=0.75)
            ax.plot(x_model, final_lc_model-(counter*offset), 'r-', lw=0.75)
            ax.text(0.01, final_lc_model[-1]-(counter*offset)+label_offset,
                    dat.split('_')[3], fontsize=10)
            ax.text(0.01, residual_offset-(counter*offset/1.5)+label_offset,
                    dat.split('_')[3], fontsize=10)
            counter += 1
        else:
            ax.scatter(ph, f-(counter*offset), c='lightgrey', marker='.', s=1)
            ax.scatter(ph-1, f-(counter*offset), c='lightgrey', marker='.', s=1)
            ax.scatter(ph, res+residual_offset-(counter*offset/1.5), c='lightgrey', marker='.', s=1)
            ax.scatter(ph-1, res+residual_offset-(counter*offset/1.5), c='lightgrey', marker='.', s=1)


    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.20, right=0.94)
    fig.savefig('{}/nites_phot_with_model.png'.format(data_dir))
