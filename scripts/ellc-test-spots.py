"""

To do:
"""
import glob as g
import ellc
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
import numpy as np

# set up a 5x5 grid of plots
def plotSpotsForParams(params, index, title):
    """
    params is a list of params to change for plotting spot variability
    index is the index in the spots array for this param set
    """
    # work in phase space
    p = np.linspace(-0.01,0.01,200)
    # set up the binary
    r_1 = 0.029544
    r_2 = 0.004616
    incl = 89.6014
    sbratio = 0
    ecc = 0.1603
    omega = 78.4526
    ld_1 = 'quad'
    ldc_1 = [0.4497, 0.1815]
    f_s = np.sqrt(ecc)*np.sin(omega*np.pi/180.)
    f_c = np.sqrt(ecc)*np.cos(omega*np.pi/180.)
    # set up a basic set of spot params, these will be cycled through
    # using the params/index input 
    spots = [[40], [-11.25], [10], [0.5]]
    # set up the plot
    fig, ax = plt.subplots(5,5,figsize=(15,10), sharex=True, sharey=True)
    fig.suptitle('Spot Variability (Lon Lat Size FR)', fontsize=20)
    c, j, k = 0, 0, 0
    for i in range(0,len(params)):
        # lon, lat, size, fr
        print(j, k, c)
        spots[index] = [params[i]]
        flux_ellc_spots = ellc.lc(p,
                                  t_zero=0,
                                  period=1,
                                  radius_1=r_1,
                                  radius_2=r_2,
                                  incl=incl,
                                  sbratio=sbratio,
                                  ld_1=ld_1,
                                  ldc_1=ldc_1,
                                  shape_1='sphere',
                                  shape_2='sphere',
                                  grid_1='default',
                                  grid_2='default',
                                  f_s=f_s,
                                  f_c=f_c,
                                  spots_1=spots)
        offset = 1 - np.average(flux_ellc_spots[:10])
        flux_ellc_no_spots = ellc.lc(p,
                                     t_zero=0,
                                     period=1,
                                     radius_1=r_1,
                                     radius_2=r_2,
                                     incl=incl,
                                     sbratio=sbratio,
                                     ld_1=ld_1,
                                     ldc_1=ldc_1,
                                     shape_1='sphere',
                                     shape_2='sphere',
                                     grid_1='default',
                                     grid_2='default',
                                     f_s=f_s,
                                     f_c=f_c)
        ax[j, k].plot(p, flux_ellc_no_spots, 'k-',
                      p, flux_ellc_spots + offset, 'r-', lw=1)
        ax[j, k].set_title('{0:.2f} {1:.3f} {2:.3f} {3:.4f}'.format(spots[0][0],
                                                                    spots[1][0],
                                                                    spots[2][0],
                                                                    spots[3][0]))
        ax[j, k].set_xlim(-0.01, 0.01)
        c += 1
        k += 1
        if c > 4:
            c = 0
            j += 1
            k = 0
    fig.savefig('spots_with_{}.png'.format(title), dpi=300)

if __name__ == "__main__":
    # set up the range of spot parameters to explore
    lon = np.linspace(-60, 60, 25)
    lat = np.linspace(-45, 45, 25)
    size = np.logspace(np.log10(0.001), np.log10(20), 25)
    fr = np.linspace(0, 3, 25)
    # plot each range
    plotSpotsForParams(lon, 0, title="varying_lon")
    plotSpotsForParams(lat, 1, title="varying_lat")
    plotSpotsForParams(size, 2, title="varying_size")
    plotSpotsForParams(fr, 3, title="varying_fr")

