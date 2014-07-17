import numpy as np
import matplotlib.pyplot as plt
from kernel_functions import gabor_kernel

# Gabor parameters
w = 0.8
phi = 0.5 * np.pi
gamma = 1 # Aspect ratio
sigma = 1
theta = 0 * np.pi

# Space parameters
dx = 0.1
lx = 6.0
dy = 0.1
ly = 6.0

xc = -1.5
yc = -1.5

xc = 1
yc = 1

Z = gabor_kernel(lx, dx, ly, dy, sigma, gamma, phi, w, theta, xc, yc)

plt.imshow(Z, extent=[-lx/2, lx/2, ly/2, -ly/2])
plt.colorbar()
#plt.show()

#plt.plot(Z[30,:])
