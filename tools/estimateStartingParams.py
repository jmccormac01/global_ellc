"""
Script to estimate the binary fit starting parameters
"""
import math

Pd = 0.854676 # days
Ms = 1.12 # Msun
Rs = 1.08 # Rsun
depth = 0.007 # %
# depth = (R2/R1)**2
# r2_a = sqrt(depth)*r1_a

Ps = Pd*24*60*60 # sec
G = 6.67408E-11 # m3 kg-1 s-2
Msun = 1988500E24 # kg
Rsun = 695700E3 # m
M1 = Ms*Msun # kg
R1 = Rs*Rsun # m

a = pow((Ps**2 * G * M1)/4*math.pi**2, (1./3.))
r1_a = R1 / a
r2_a = math.sqrt(depth)*r1_a

print('a: {} m'.format(a))
print('r1_a : {}'.format(r1_a))
print('r2_a : {}'.format(r2_a))
print('a_r1 : {}'.format(1./r1_a))
