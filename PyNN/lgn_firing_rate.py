"""
This function allow us to visualize the result firing rate from a lgn cell. The objective of this functions is to 
reproduce figure 1 in Troyer paper. 
"""

import numpy as np
from misc_functions import create_standar_indexes, create_extra_indexes
from stimuli_functions import sine_grating, ternary_noise 
from kernel_functions import create_kernel 
from analysis_functions import convolution 
import matplotlib.pyplot as plt 
from plot_functions import visualize_firing_rate

## Time parameters

# Time scales   
dt = 0.25  # ms
dt = 1.0
dt_kernel = 5.0  # ms
dt_stimuli = 5.0  # ms
kernel_duration = 150  # ms
kernel_size = int(kernel_duration / dt_kernel)


# Simulation time duration 
T_simulation = 5 * 10 ** 2.0 # ms
T_simulation += int(kernel_size * dt_kernel)  # Add the size of the kernel
Nt_simulation = int(T_simulation / dt)  # Number of simulation points 
N_stimuli = int(T_simulation / dt_stimuli)  # Number of stimuli points 

## Space parameters

# visual space resolution and size  
dx = 0.05
dy = 0.05
lx = 6.0  # In degrees
ly = 6.0  # In degrees

# center-surround parameters 
factor = 1  # Controls the overall size of the center-surround pattern
sigma_center = 0.25 * factor  # Corresponds to 15'
sigma_surround = 1 * factor  # Corresponds to 1 degree


# sine grating spatial parameters 
K = 0.8  # Cycles per degree 
Phi = 0 * np.pi
Theta = 0 * np.pi
max_contrast = 2.0 * 2
contrast = 0.5  # Percentage
A = contrast * max_contrast 
# Temporal frequency of sine grating 
w = 3  # Hz

# Set the random set for reproducibility
seed = 1053
np.random.seed(seed)

# Create indexes 
signal_indexes, delay_indexes, stimuli_indexes = create_standar_indexes(dt, dt_kernel, dt_stimuli, kernel_size, Nt_simulation)
working_indexes, kernel_times = create_extra_indexes(kernel_size, Nt_simulation)

# Initialize the signal array to be filled 
firing_rate_on = np.zeros(Nt_simulation)
firing_rate_off = np.zeros(Nt_simulation)

# Create the stimuli 
# stimuli = ternary_noise(N_stimuli, int(lx /dx), int(ly/dt))
stimuli = sine_grating(dx, lx, dy, ly, A, K, Phi, Theta, dt_stimuli, N_stimuli, w)

# Chose the particular cell 
xc = 0
yc = 0 

# Create the kernel 
kernel_on = create_kernel(dx, lx, dy, ly, sigma_surround, sigma_center, dt_kernel, kernel_size, inverse=1, x_tra=0, y_tra=0)
kernel_off = create_kernel(dx, lx, dy, ly, sigma_surround, sigma_center, dt_kernel, kernel_size, inverse=-1, x_tra=0, y_tra=0)

# Calculate the firing rate 
for index in signal_indexes:
    firing_rate_on[index] = convolution(index, kernel_times, delay_indexes, stimuli_indexes, kernel_on, stimuli)

for index in signal_indexes:
    firing_rate_off[index] = convolution(index, kernel_times, delay_indexes, stimuli_indexes, kernel_off, stimuli)

# Add background noise 
on_noise = 10  # Hz
off_noise = 15  # HZ

firing_rate_on += on_noise
firing_rate_off += off_noise

# Rectify firing rates 
firing_rate_on[firing_rate_on < 0] = 0
firing_rate_off[firing_rate_off < 0] = 0

# Plot the firing rates 
visualize_firing_rate(firing_rate_on[signal_indexes], dt,
                      T_simulation - int(kernel_size * dt_kernel), label='Firing_rate_on')
visualize_firing_rate(firing_rate_off[signal_indexes], dt,
                      T_simulation - int(kernel_size * dt_kernel), label='Firing_rate_off')
plt.ylim((0, 60))
plt.show()