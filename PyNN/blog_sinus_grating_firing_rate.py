"""
Code for the blog post in July about the convolution of the center-surround kernel with the sinusoidal grating stimuli
"""

import numpy as np
from kernel_functions import create_kernel
from stimuli_functions import sine_grating
from misc_functions import create_standar_indexes, create_extra_indexes
from analysis_functions import convolution
import matplotlib.pyplot as plt

# First we define the size and resolution of the space in which the convolution is going to happen
dx = 0.05
dy = 0.05
lx = 6.0  # In degrees
ly = 6.0  # In degrees

# Now we define the temporal parameters of the kernel
dt_kernel = 5.0  # ms
kernel_duration = 150  # ms
kernel_size = int(kernel_duration / dt_kernel)

#  Now the center surround parameters
factor = 1  # Controls the overall size of the center-surround pattern
sigma_center = 0.25 * factor  # Corresponds to 15'
sigma_surround = 1 * factor  # Corresponds to 1 degree

# Finally we create the kernel
kernel_on = create_kernel(dx, lx, dy, ly, sigma_surround, sigma_center, dt_kernel, kernel_size)

## Now we define the temporal l parameters of the sinus grating
dt_stimuli = 5.0  # ms

# We also need to add how long do we want to convolve
dt = 1.0  # Simulation resolution
T_simulation = 1 * 10 ** 3.0  # ms
T_simulation += int(kernel_size * dt_kernel)  # Add the size of the kernel
Nt_simulation = int(T_simulation / dt)  # Number of simulation points
N_stimuli = int(T_simulation / dt_stimuli)  # Number of stimuli points

# And now the spatial parameters of the sinus grating
K = 0.8  # Cycles per degree
Phi = 0
Theta = 0
max_contrast = 2.0 * 2
contrast = 0.5  # Percentage
A = contrast * max_contrast
# Temporal frequency of sine grating
w = 3  # Hz

stimuli = sine_grating(dx, lx, dy, ly, A, K, Phi, Theta, dt_stimuli, N_stimuli, w)


## Now we can do the convolution

# First we define the necessary indexes to the convolution
signal_indexes, delay_indexes, stimuli_indexes = create_standar_indexes(dt, dt_kernel, dt_stimuli, kernel_size, Nt_simulation)
working_indexes, kernel_times = create_extra_indexes(kernel_size, Nt_simulation)

# Now we calculate the signal
signal = np.zeros(Nt_simulation)

for index in signal_indexes:
    signal[index] = convolution(index, kernel_times, delay_indexes, stimuli_indexes, kernel_on, stimuli)

#Plot the signal
t = np.arange(kernel_size*dt_kernel, T_simulation, dt)
plt.plot(t, signal[signal_indexes])
plt.show()