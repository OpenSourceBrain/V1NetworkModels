import numpy as np
import matplotlib.pyplot as plt 
from kernel_functions import spatial_kernel

dx = 0.1
dy = 0.1

lx = 6.0 # In degrees 
ly = 6.0 # In degrees 

factor = 2

xc = 0
yc = 0
sign = 1 # On or Off 

sigma_center = 0.25 * factor # Corresponds to 15' original
sima_center = 0.40 * factor # Allen as in Mozaik 
#sigma_center = 0.15 * factor 
sigma_surround = 1 * factor  # Corresponds to 1 degree
theta = 0


Z = spatial_kernel(lx, dx, ly, dy, sigma_center, sigma_surround, inverse=sign, x_tra=xc, y_tra=yc, theta=90)

plt.imshow(Z, extent=[-lx/2,lx/2,ly/2,-ly/2])
plt.colorbar()
plt.show()