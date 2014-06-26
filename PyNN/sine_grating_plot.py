"""
This functions is just to visualize the sine-grating and provide and idea and context of the 
scale and overall shape of the pattern 
"""

import numpy as np
import matplotlib.pyplot as plt 
from stimuli_functions import sine_grating 

# Temporal pattern
dt_stimuli = 1.0 # in ms 
N_stimuli = 100

# Space parameters 
dx = 0.05
dy = 0.05
lx = 5.0 # In degrees 
ly = 5.0 # In degrees

# sine grating spatial parameters 
K = 1.0 # cycles per degree 
Phi = 0 * np.pi
Theta = 0 * np.pi
A = 1.0
# Temporal frequency of sine grating 
w = 3 # Hz 

stimuli = sine_grating(dx, lx, dy, ly, A, K, Phi, Theta, dt_stimuli, N_stimuli, w)

Z = stimuli[0,...]

plt.imshow(Z, extent=[-lx/2,lx/2,ly/2,-ly/2])
plt.colorbar()
plt.show()

# x = np.arange(-lx/2,lx/2,dx)
# y = Z[0,:]
# plt.plot(x,y)
# plt.show()
# 
# t = np.arange(0, N_stimuli * dt_stimuli, dt_stimuli)
# y = stimuli[:,0,0]
# plt.plot(t,y)
# plt.show()