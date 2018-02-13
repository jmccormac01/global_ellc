import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, Column

chain = Table.read('chain.csv')
chain = Table(chain, masked=True)
nwalkers = 1+np.max(chain['walker'])
nsteps = 1+np.max(chain['step'])
print('Read chain.csv with {:d} walkers of {:d} steps'
    .format(nwalkers,nsteps))

maxlike=np.empty(nwalkers)
for i in range(0,nwalkers):
    n=np.where(chain['walker']==i)
    maxlike[i] = max(chain['loglike'][n])
    print(i, maxlike[i])
    plt.plot(chain['loglike'][n],'.')
plt.show()

n2 = np.where(maxlike == max(maxlike))[0][0]
print('Walker {0:d} has maximum likelihood {1:.2f}'.format(n2, maxlike[n2]))

# now plot the params for the best walker
for i in chain.colnames:
    print('plotting {0:s} for walker {1:d}'.format(i, n2))
    n3 = np.where(chain['walker']==n2)
    plt.plot(chain[i][n3],'r.')
    plt.title(i)
    plt.show()

