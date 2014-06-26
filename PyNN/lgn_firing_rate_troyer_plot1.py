"""
This function allow us to visualize the result firing rate from a lgn cell. The objective of this functions is to 
reproduce figure 1 in Troyer paper. 
"""

import numpy as np
from misc_functions import create_standar_indexes, create_extra_indexes
from stimuli_functions import sine_grating, ternary_noise 
from kernel_functions import create_kernel 
#from analysis_functions import produce_spikes, convolution
from analysis_functions import convolution 
import matplotlib.pyplot as plt 
from plot_functions import visualize_firing_rate

## Time parameters

# Time scales   
dt = 0.25  # ms
dt = 1.0 
dt_kernel = 1.0  # ms
dt_stimuli = 1.0  # ms
kernel_duration = 150  # ms
kernel_size = int(kernel_duration / dt_kernel)

# Simulation time duration 
T_simulation = 6 * 10 ** 2.0  # ms
T_simulation += int(kernel_size * dt_kernel)  # Add the size of the kernel
Nt_simulation = int(T_simulation / dt)  # Number of simulation points 
N_stimuli = int(T_simulation / dt_stimuli)  # Number of stimuli points 

## Space parameters

# visual space resolution and size  
dx = 0.1
dy = 0.1
lx = 6.0  # In degrees
ly = 6.0  # In degrees

# center-surround parameters 
factor = 1  # Controls the overall size of the center-surround pattern
sigma_center = 0.25 * factor # Corresponds to 15' 
sigma_surround = 1 * factor  # Corresponds to 1 degree


# sine grating spatial parameters 
K = 0.8  # Cycles per degree 
Phi = 0 * np.pi
Theta = 0 * np.pi
max_contrast = 2.0 * 2
contrast1 = 0.5  # Percentage
A1 = contrast1 * max_contrast
contrast2 = 0.025  # Percentage
A2 = contrast2 * max_contrast 
# Temporal frequency of sine grating 
w = 3  # Hz

# Set the random set for reproducibility
seed = 1053
np.random.seed(seed)

# Create indexes 
signal_indexes, delay_indexes, stimuli_indexes = create_standar_indexes(dt, dt_kernel, dt_stimuli, kernel_size, Nt_simulation)
working_indexes, kernel_times = create_extra_indexes(kernel_size, Nt_simulation)

# Initialize the signal array to be filled 
firing_rate_on1 = np.zeros(Nt_simulation)
firing_rate_off1 = np.zeros(Nt_simulation)

firing_rate_on2 = np.zeros(Nt_simulation)
firing_rate_off2 = np.zeros(Nt_simulation)

# Create the stimuli
stimuli1 = sine_grating(dx, lx, dy, ly, A1, K, Phi, Theta, dt_stimuli, N_stimuli, w)
stimuli2 = sine_grating(dx, lx, dy, ly, A2, K, Phi, Theta, dt_stimuli, N_stimuli, w)

# Chose the particular cell
xc = 0
yc = 0 

# Create the kernels
kernel_on = create_kernel(dx, lx, dy, ly, sigma_surround, sigma_center,
                          dt_kernel, kernel_size, inverse=1, x_tra=0, y_tra=0)
kernel_off = create_kernel(dx, lx, dy, ly, sigma_surround, sigma_center,
                           dt_kernel, kernel_size, inverse=-1, x_tra=0, y_tra=0)

# Calculate the firing rate through convolution

for index in signal_indexes:
    firing_rate_on1[index] = convolution(index, kernel_times, delay_indexes, stimuli_indexes, kernel_on, stimuli1)

for index in signal_indexes:
    firing_rate_off1[index] = convolution(index, kernel_times, delay_indexes, stimuli_indexes, kernel_off, stimuli1)

for index in signal_indexes:
    firing_rate_on2[index] = convolution(index, kernel_times, delay_indexes, stimuli_indexes, kernel_on, stimuli2)

for index in signal_indexes:
    firing_rate_off2[index] = convolution(index, kernel_times, delay_indexes, stimuli_indexes, kernel_off, stimuli2)

# Add background noise 
on_noise = 10  # Hz
off_noise = 15  # Hz

firing_rate_on1 += on_noise
firing_rate_off1 += off_noise

firing_rate_on2 += on_noise
firing_rate_off2 += off_noise

# Rectify firing rates 
firing_rate_on1[firing_rate_on1 < 0] = 0
firing_rate_off1[firing_rate_off1 < 0] = 0

firing_rate_on2[firing_rate_on2 < 0] = 0
firing_rate_off2[firing_rate_off2 < 0] = 0

remove_start = int(kernel_size * dt_kernel)

# Plot the firing rates 
plt.subplot(2, 1, 1)
# Plot first the on-center firing rates 
plt.title('ON-center')

# Plot background levels 
background = np.ones(signal_indexes.size) * on_noise
plt.plot(signal_indexes - remove_start, background, '-', color='black', label='Background noise ')
# Plot the firing rates p
visualize_firing_rate(firing_rate_on1[signal_indexes], dt, T_simulation - remove_start, label='50  % contrast')
visualize_firing_rate(firing_rate_on2[signal_indexes], dt, T_simulation - remove_start, label='2.5 % contrast')
plt.ylim((0, 60))

plt.subplot(2, 1, 2)
# Now we plot the off-center firing rates
plt.title('OFF-center')
# Plot the background noise level 
background = np.ones(signal_indexes.size) * off_noise
plt.plot(signal_indexes - remove_start, background, '-', color='black', label='Background noise ')
# Plot the firing rates 
visualize_firing_rate(firing_rate_off1[signal_indexes], dt, T_simulation - remove_start, label='50  % contrast')
visualize_firing_rate(firing_rate_off2[signal_indexes], dt, T_simulation - remove_start, label='2.5 % contrast')
plt.ylim((0, 60))
plt.show()