"""
To do:
    add another set of subplots for the data-model residuals
"""
import glob as g
import ellc
import matplotlib.pyplot as plt
import numpy as np

E = 2456457.8905076
P = 16.9535269
rp = 0.165
a = 35.2016136 #34.20161364
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
data_dir = '/Users/James/Documents/EBLMS/J23431841/'
filelist = g.glob('*J234318.41*.lc.txt')

# generate list of epochs
epochs = np.empty(1000)
for i in range(0, len(epochs)):
    epochs[i] = E + i*P

fig, ax = plt.subplots(len(filelist), figsize=(10,10))
c = 0
for data_file in filelist:
    input_file = '{0:s}{1:s}'.format(data_dir, data_file)
    t, f, e = np.loadtxt(input_file, usecols=[2, 3, 4], unpack=True)

    diff = abs(epochs - np.average(t))
    diff_min = np.where(diff == min(diff))
    t_zero = epochs[diff_min][0]
    print(t_zero)
    p=(t-t_zero)/P

    flux_ellc_d = ellc.lc(p, t_zero=0,period=1,radius_1=r_1,
        radius_2=r_2,incl=incl,sbratio=sbratio,
        ld_1=ld_1, ldc_1=ldc_1,shape_1='sphere',shape_2='sphere',
        grid_1='default',grid_2='default', f_s=f_s, f_c=f_c)

    ax[c].plot(p, f, 'r.')
    ax[c].plot(p, flux_ellc_d, 'k--', lw=3)
    ax[c].set_ylim(0.96, 1.01)
    #ax[c].set_xlim(-0.02,0.02)
    c+=1

plt.show()
