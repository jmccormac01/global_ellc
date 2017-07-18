import uncertainties as uc

R1 = uc.ufloat(0.854, 0.06)
R2 = uc.ufloat(0.127, 0.007)
a = uc.ufloat(0.124, 0.0066)

Rsun = 695700E3
AU = 1.4960E11

r_1 = (R1*Rsun)/(a*AU)
r_2 = (R2*Rsun)/(a*AU)
aR1 = (a*AU)/(R1*Rsun)

print("R1/a = {}".format(r_1))
print("R2/a = {}".format(r_2))
print("a/R1 = {}".format(aR1))


