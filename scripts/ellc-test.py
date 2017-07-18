"""
To do:
    add another set of subplots for the data-model residuals
"""
import matplotlib
matplotlib.use('QT5Agg')
import glob as g
import ellc
import matplotlib.pyplot as plt
import numpy as np

E = 2456457.8905076
P = 16.9535269
rp = 0.165
a = 34.20161364
incl = 89.75000
ecc = 0.16
w = 74.404
ldc_1 = [0.1, 0.3]
ld_1 = 'quad'
sbratio = 0

r_1 = 1./a
r_2 = r_1*rp

f_s = np.sqrt(ecc)*np.sin(w*np.pi/180.)
f_c = np.sqrt(ecc)*np.cos(w*np.pi/180.)

# import data and work out t_zero
data_dir = '/Users/jmcc/Dropbox/EBLMS/J23431841/'
filelist = g.glob('{}/NITES_J234318.41_20120829_Clear_F1*'.format(data_dir))

# lon, lat, size, fr
spot_config = [[40], [-11.25], [10], [0.5]]

# generate list of epochs
epochs = np.empty(1000)
for i in range(0, len(epochs)):
    epochs[i] = E + i*P

fig, ax = plt.subplots(len(filelist)+1, figsize=(10,10))
c = 0
for data_file in filelist:
    t, f, e = np.loadtxt(data_file, usecols=[2, 3, 4], unpack=True)

    diff = abs(epochs - np.average(t))
    diff_min = np.where(diff == min(diff))
    t_zero = epochs[diff_min][0]
    print(t_zero)
    p=(t-t_zero)/P

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
                        spots_1=spot_config)

    ax[c].plot(p, f, 'r.')
    ax[c].plot(p, flux_ellc_d, 'k--', lw=3)
    #ax[c].set_ylim(0.96, 1.01)
    #ax[c].set_xlim(-0.02,0.02)
    c+=1

plt.show()
