import numpy as np
import matplotlib.pyplot as plt
from kernel_functions import gabor_kernel
from connector_functions import gabor_probability
from scipy.stats.stats import pearsonr


# Gabor parameters
w = 0.8
phi = 0
gamma = 1  # Aspect ratio
sigma = 1
theta = 0


# Space parameters
dx = 0.1
lx = 6.0
dy = 0.1
ly = 6.0

xc = 0
yc = 0

size_x = 0.75
size_y = 0.75

x_values = np.arange(-size_x/2, size_x/2, dx)
y_values = np.arange(-size_y/2, size_y/2, dy)

points = 50

phis = np.linspace(-180, 180, points)
thetas = np.linspace(-90, 90, points)

correlation_phi = np.zeros(phis.size)
correlation_theta = np.zeros(thetas.size)
correlation_x = np.zeros(x_values.size)
correlation_y = np.zeros(y_values.size)

Z1 = gabor_kernel(lx, dx, ly, dy, sigma, gamma, phi, w, theta, xc, yc)

# Phase

for index, phi2 in enumerate(phis):
    Z2 = gabor_kernel(lx, dx, ly, dy, sigma, gamma, phi2, w, theta, xc, yc)

    correlation = pearsonr(Z1.flat, Z2.flat)[0]
    correlation_phi[index] = correlation

# Orientations

for index, theta2 in enumerate(thetas):
    Z2 = gabor_kernel(lx, dx, ly, dy, sigma, gamma, phi, w, theta2, xc, yc)

    correlation = pearsonr(Z1.flat, Z2.flat)[0]
    correlation_theta[index] = correlation

# X's
for index, x2 in enumerate(x_values):
    Z2 = gabor_kernel(lx, dx, ly, dy, sigma, gamma, phi, w, theta, x2, yc)

    correlation = pearsonr(Z1.flat, Z2.flat)[0]
    correlation_x[index] = correlation

# Y's
for index, y2 in enumerate(y_values):
    Z2 = gabor_kernel(lx, dx, ly, dy, sigma, gamma, phi, w, theta, xc, y2)

    correlation = pearsonr(Z1.flat, Z2.flat)[0]
    correlation_y[index] = correlation

plt.subplot(2, 2, 1)
plt.title('Correlation with varying phi')
plt.plot(phis, correlation_phi)

plt.subplot(2, 2, 2)
plt.title('Correlation with varying theta')
plt.plot(thetas, correlation_theta)

plt.subplot(2, 2, 3)
plt.title('Correlation with varying x')
plt.plot(x_values, correlation_x)

plt.subplot(2, 2, 4)
plt.title('Correlation with varying y')
plt.plot(y_values, correlation_y)

plt.show()