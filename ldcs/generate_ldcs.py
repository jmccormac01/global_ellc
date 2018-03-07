"""
Code to generate LDCs using LDTK

Current values are for J234318.41
"""
from ldtk.filters import sdss_r
from ldtk import LDPSetCreator

# J234318.41
# Teff 6400, 50
# logg 4.40, 0.17
# z 0.27, 0.11

if __name__ == "__main__":
    sc = LDPSetCreator(teff=(5963, 139),
                       logg=(4.42, 0.25),
                       z=(0.00, 0.2),
                       filters=[sdss_r])
    ps = sc.create_profiles()
    cq, eq = ps.coeffs_qd(do_mc=True)
    print('LDC_1 {:.4f} {:.4f}'.format(cq[0][0], eq[0][0]))
    print('LDC_2 {:.4f} {:.4f}'.format(cq[0][1], eq[0][1]))

