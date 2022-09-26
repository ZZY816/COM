import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw

x = np.linspace(0, 1, 1000)

#
# e = np.ones(x.shape)*(-2/np.exp(1.))
# tau = np.log(10)
# lam = 1
# z = 0.5*np.maximum(e, (x-tau)/lam)
# sigma = np.exp(-lambertw(z))
# y = (x - tau)*sigma + lam*(np.log(sigma)**2)

threshold = 0.5
elongation = -10
height = 1.2
height2 = 0.2


y = height / (1 + np.exp(elongation*(x-threshold))) + 1 - height/2
z = height2 / (1 + np.exp(elongation*(x-threshold))) + 1 - height2/2
#plt.figure(figsize=(6,4))
plt.plot(x, y, color="red", linewidth=1)
plt.plot(x, z, color="blue", linewidth=1)
plt.show()