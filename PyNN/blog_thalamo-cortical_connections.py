from pyNN.utility import get_script_args, Timer
import numpy as np
import matplotlib.pyplot as plt
from connector_functions import gabor_probability
from kernel_functions import gabor_kernel

n_pick = 5
g = 3.0

w = 0.8
phi = 0
gamma = 1  # Aspect ratio
sigma = 1
theta = 0

# Space parameters
dx = 0.1
lx = 3.0
dy = 0.1
ly = 3.0

xc = 0
yc = 0

x_values = np.arange(-lx/2, lx/2, dx)
y_values = np.arange(-ly/2, ly/2, dy)

polarity = 1.0

Z = np.zeros((x_values.size, y_values.size))
L = np.zeros(Z.shape)

for x_index, x in enumerate(x_values):
    for y_index, y in enumerate(y_values):
        probability = polarity * gabor_probability(x, y, sigma, gamma, phi, w, theta, xc, yc)
        print 'proba', probability
        counts = np.random.rand(n_pick) < probability
        print 'counts', counts
        aux = np.sum(counts)  # Samples
        synaptic_weight = (g / n_pick) * aux
        L[x_index, y_index] = aux
        Z[x_index, y_index] = synaptic_weight
        print x, y, synaptic_weight, probability
        print '------------'

plt.subplot(1, 2, 1 )
plt.imshow(Z.transpose(), extent=[-lx/2, lx/2, ly/2, -ly/2])


plt.subplot(1, 2, 2)
Z = gabor_kernel(lx, dx, ly, dy, sigma, gamma, phi, w, theta, xc, yc)

plt.imshow(Z, extent=[-lx/2, lx/2, ly/2, -ly/2])


plt.show()