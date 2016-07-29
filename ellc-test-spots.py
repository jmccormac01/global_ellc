"""
To do:
"""
import glob as g
import ellc
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
import numpy as np

# simulate an eblm
rp = 0.165
a = 34.20161364
incl = 89.75000
ecc = 0.16
w = 74.404
ldc_1 = [0.1, 0.3]
ld_1 = 'quad'
sbratio = 0

# set up some number for ellc
r_1 = 1./a
r_2 = r_1*rp
f_s = np.sqrt(ecc)*np.sin(w*np.pi/180.)
f_c = np.sqrt(ecc)*np.cos(w*np.pi/180.)

# set up the range of spot parameters to explore
lon = np.linspace(-60, 60, 25)
lat = np.linspace(-45, 45, 25)
size = np.logspace(np.log10(0.001), np.log10(20), 25)
fr = np.linspace(0, 1, 25)

# work in phase space
p = np.linspace(-0.01,0.01,200)

# set up a 5x5 grid of plots
def plotSpotsForParams(params, index):
    """
    params is a list of params to change for plotting spot variability
    index is the index in the spots array for this param set
    """
    fig, ax = plt.subplots(5,5,figsize=(15,10), sharex=True, sharey=True)
    fig.suptitle('Spot Variability (Lon Lat Size FR)', fontsize=20)
    c, j, k = 0, 0, 0
    for i in range(0,len(params)):
        # lon, lat, size, fr
        print(j, k, c)
        spots[index] = [params[i]]
        flux_ellc_d = ellc.lc(p,
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
        ax[j, k].plot(p, flux_ellc_d, 'k-', lw=1)
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
    plt.show()

spots = [[40], [-11.25], [10], [0.5]]
plotSpotsForParams(lon,0)
spots = [[40], [-11.25], [10], [0.5]]
plotSpotsForParams(lat,1)
spots = [[40], [-11.25], [10], [0.5]]
plotSpotsForParams(size,2)
spots = [[40], [-11.25], [10], [0.5]]
plotSpotsForParams(fr,3)

