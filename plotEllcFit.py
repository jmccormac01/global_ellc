import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, Column

fig = plt.figure(1, figsize=(10, 5))

model = Table.read('model.csv')
model = Table(model, masked=True)
mask = model['flag'] < 0 

ph_obs = np.extract(mask == False,model['phase']) 
m_obs = np.extract(mask == False,model['mag'])
m_fit = np.extract(mask == False,model['fit'])
m_res = m_obs - m_fit
print ('RMS residual = {0:0.3f}'.format(1000*np.std(m_res)))
ph_plt = np.array(model['phase']) 
f_plt = np.array(model['fit'])
i_sort = np.argsort(ph_plt)
ph_plt = ph_plt[i_sort]
f_plt  = f_plt[i_sort]

plt.scatter(ph_obs,m_obs,color='darkgreen',marker='x',s=3)
plt.scatter(ph_obs-1,m_obs,color='darkgreen',marker='x',s=3)
plt.plot(ph_plt,f_plt,color='darkblue')
plt.plot(ph_plt-1,f_plt,color='darkblue')
plt.xlim([-0.02,0.02])
plt.ylim([0.05,-0.02])
plt.show()
