import uncertainties as uc

R1 = uc.ufloat(0.854, 0.06)
R2 = uc.ufloat(0.127, 0.007)
a = uc.ufloat(0.127, 0.0066)

Rsun = 695700E3
AU = 1.4960E11

r_1 = (R1*Rsun)/(a*AU)
r_2 = (R2*Rsun)/(a*AU)

print(r_1)
print(r_2)


