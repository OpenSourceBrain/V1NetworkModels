"""
This function allow us to visualize the result firing rate from a lgn cell that we can choose and then 
visualize the spikes produce by it 
"""

import numpy as np
from misc_functions import create_standar_indexes, create_extra_indexes
from stimuli_functions import sine_grating, ternary_noise 
from kernel_functions import create_kernel 
from analysis_functions import produce_spikes, convolution
import matplotlib.pyplot as plt 
from plot_functions import visualize_firing_rate

## Time parameters

# Time scales   
dt = 0.25  # milliseconds
dt = 1
dt_kernel = 1.0  # ms
dt_stimuli = 1.0  # ms
kernel_duration = 150  # ms
kernel_size = int(kernel_duration / dt_kernel)

# Simulation time duration
T_simulation = 2 * 10 ** 3.0  # ms
remove_start = int(kernel_size * dt_kernel)
T_simulation += remove_start  # Add the size of the kernel
Nt_simulation = int(T_simulation / dt)  # Number of simulation points 
N_stimuli = int(T_simulation / dt_stimuli)  # Number of stimuli points 

## Space parameters
# visual space resolution and size  
dx = 0.1
dy = 0.1
lx = 6.8  # In degrees
ly = 6.8  # In degrees

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
seed = 31255433
np.random.seed(seed)

# Create indexes 
signal_indexes, delay_indexes, stimuli_indexes = create_standar_indexes(dt, dt_kernel,
                                                                        dt_stimuli, kernel_size, Nt_simulation)
working_indexes, kernel_times = create_extra_indexes(kernel_size, Nt_simulation)

# Initialize the signal array to be filled 
firing_rate = np.zeros(Nt_simulation)

# Create the stimuli 
stimuli = sine_grating(dx, lx, dy, ly, A, K, Phi, Theta, dt_stimuli, N_stimuli, w)

# Chose the particular cell 
xc = 1.5
yc = 1.5

# Create the kernel 
kernel = create_kernel(dx, lx, dy, ly, sigma_surround, sigma_center,
                       dt_kernel, kernel_size, inverse=1, x_tra=xc, y_tra=yc)

# Calculate the firing rate 
for index in signal_indexes:
    firing_rate[index] = convolution(index, kernel_times, delay_indexes, stimuli_indexes, kernel, stimuli)


firing_rate += 10 # Add background noise 
# Rectify the firing rate
firing_rate[ firing_rate < 0] = 0

remove_start = int(kernel_size *dt_kernel)
visualize_firing_rate(firing_rate[signal_indexes], dt, T_simulation - remove_start, label='Firing rate')
#visualize_firing_rate(firing_rate / np.max(firing_rate), dt, T_simulation, label='Firing rate')

# Produce spikes with the signal
spike_times_thin = produce_spikes(firing_rate, dt, T_simulation, remove_start)
spike_times_thin -= remove_start 
 
y = np.ones_like(spike_times_thin) * np.max(firing_rate) * 0.5
plt.plot(spike_times_thin, y, '*', label='spikes')
plt.legend()
plt.ylim([0,60])
plt.show()
