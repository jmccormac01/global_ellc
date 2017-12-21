"""
Code to take some known model params
and plot individual wasp transits and
residuals with the model
"""
from collections import defaultdict
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
from matplotlib import rc, cycler
import numpy as np
import ellc

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
       prop_cycle=cycler('color',['black']))
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
nights_to_plot = [3219, 3236, 3253, 3270, 3948, 4050, 4067, 4321]
data_dir = "/Users/jmcc/Dropbox/EBLMs/J23431841"
data_file = "1SWASPJ234318.41+295556.5_OF2328+3237_100_ORCA_TAMTFA.lc"
infile = "{}/{}".format(data_dir, data_file)

t, m, e = np.loadtxt(infile, usecols=[0, 1, 2], unpack=True)
tn, mn, en = split_into_nights(t, m, e)

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


one_column()
general()
fig, ax = plt.subplots(1)
ax.set_xlim(-0.02, 0.02)
ax.set_xlabel('Orbital phase')
ax.set_ylabel('Flux ratio')
offset = 0.05
counter = 0
for night in tn:
    if night in nights_to_plot:
        ph = (((np.array(tn[night]) + 2450000) - t0)/period)%1
        # convert back to flux ratio to match ellc models
        f = pow(10, (np.array(mn[night])/-2.5))
        fe = (f * np.array(en[night]))/2.5*np.log(10)
        ax.scatter(ph, f-(counter*offset), c='k', marker='.', s=7)
        ax.scatter(ph-1, f-(counter*offset), c='k', marker='.', s=7)
        ax.plot(x_model, final_lc_model-(counter*offset), 'r-', lw=0.75)
        ax.text(0.01, final_lc_model[-1]-(counter*offset)+0.005, str(night), fontsize=10)
        counter += 1
    else:
        continue
fig.subplots_adjust(top=0.95, bottom=0.15, left=0.16, right=0.94)
fig.savefig('{}/wasp_phot_with_model.png'.format(data_dir))
