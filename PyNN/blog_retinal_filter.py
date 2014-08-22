import numpy as np
import matplotlib.pyplot as plt
from plot_functions import plot_mutliplot_bilinear


dx = 0.1  # The resolution in x
dy = 0.1  # The resolution in y

lx = 50 # How long are kernels in the x direction
ly = 50 # The long are kernels in the y direction

# Create the positions
x = np.arange(-lx/2, lx/2, dx)
y = np.arange(-ly/2, ly/2, dy)

sigma_center = 15  # Size of the center area
sigma_surround = 7  # Size of the surround area

# This is for the two dimensional pattern
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)  # Distance
center = (17.0 / sigma_center**2) * np.exp(-(R / sigma_center)**2)
surround = (16.0 / sigma_surround**2) * np.exp(-(R / sigma_surround)**2)
Z = surround - center 

# Plot countour map 
plt.contourf(X, Y, Z, 50, alpha=0.75, cmap=plt.cm.hot)
plt.colorbar()
 
C = plt.contour(X, Y, Z, 10, colors='black', linewidth=.5)
plt.clabel(C, inline=10, fontsize=10)
plt.show()

# One dimensionall side view 
center = (17.0 / sigma_center**2) * np.exp(-(x / sigma_center)**2)
surround = surround = (16.0 / sigma_surround**2) * np.exp(-(x / sigma_surround)**2)
z1 = surround - center 

plt.plot(x,z1)

plt.show()

##  Now we code the temporal pattern 

# First the kernel size and resolution 
kernel_size = 25
dt_kernel = 10
t = np.arange(0, kernel_size * dt_kernel, dt_kernel) # Time vector 

## Temporal parameters
K1 = 1.05
K2 = 0.7
c1 = 0.14
c2 = 0.12
n1 = 7.0
n2 = 8.0
t1 = -6.0
t2 = -6.0
td = 6.0

p1 = K1 * ((c1*(t - t1))**n1 * np.exp(-c1*(t - t1))) / ((n1**n1) * np.exp(-n1))
p2 = K2 * ((c2*(t - t2))**n2 * np.exp(-c2*(t - t2))) / ((n2**n2) * np.exp(-n2))
p3 = p1 - p2


plt.plot(t, p3, label='temporal kernel')
plt.xlabel('time (ms)')
plt.legend()
plt.show()


## Now create the spatio-temporal filter 

# Initialize and fill the spatio-temporal kernel  
kernel = np.zeros((kernel_size, int(lx/dx), int(ly/dy)))

for k, p in enumerate(p3):
    kernel[k,...] = p * Z
    
plot_mutliplot_bilinear(25,kernel, colorbar=True, symmetric=2)
p

plt.show()

