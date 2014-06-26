"""
This function creates and stores the LGN spikes patterns. It creates and stores them in an order that is consistent 
with the one that PyNN uses to create the spike arrays. That is, for each x it will first sweep completely the y coordaintes
and the it will change x 
"""

import numpy as np

from misc_functions import create_standar_indexes, create_extra_indexes
from stimuli_functions import sine_grating
from kernel_functions import create_kernel
from analysis_functions import produce_spikes, convolution
from plot_functions import visualize_firing_rate


## Time parameters

# Time scales
dt = 0.25  # milliseconds
dt = 1
dt_kernel = 5.0  # ms
dt_stimuli = 5.0  # ms
kernel_duration = 150  # ms
kernel_size = int(kernel_duration / dt_kernel)

# Simulation time duration
T_simulation = 5 * 10 ** 2.0  # ms
remove_start = int(kernel_size * dt_kernel)
T_simulation += remove_start  # Add the size of the kernel
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
seed = 1055
np.random.seed(seed)

# Create indexes
signal_indexes, delay_indexes, stimuli_indexes = create_standar_indexes(dt, dt_kernel,
                                                                        dt_stimuli, kernel_size, Nt_simulation)
working_indexes, kernel_times = create_extra_indexes(kernel_size, Nt_simulation)

# Initialize the signal array to be filled
firing_rate = np.zeros(Nt_simulation)

# Create the stimuli
# stimuli = ternary_noise(N_stimuli, int(lx /dx), int(ly/dt))
stimuli = sine_grating(dx, lx, dy, ly, A, K, Phi, Theta, dt_stimuli, N_stimuli, w)

# Create the cell array
x_values = np.arange(-(lx-1)/2, (lx-1)/2, dx)
y_values = np.arange(-(ly-1)/2, (ly-1)/2, dy)

spike_train = []
positions =[]

for x in x_values:
    for y in y_values:
        print '----------'
        print 'Creating spikes'
        print 'Order of creation in the grid'
        print 'x=', x
        print 'y=', y

        xc = x
        yc = y

        positions.append((x,y))


        # Create the kernel
        kernel = create_kernel(dx, lx, dy, ly, sigma_surround, sigma_center,
                               dt_kernel, kernel_size, inverse=1, x_tra=xc, y_tra=yc)

        # Calculate the firing rate
        for index in signal_indexes:
            firing_rate[index] = convolution(index, kernel_times, delay_indexes, stimuli_indexes, kernel, stimuli)

        firing_rate += 10  # Add background noise
        # Rectify the firing rate
        firing_rate[ firing_rate < 0] = 0
        #
        visualize_firing_rate(firing_rate / np.max(firing_rate), dt, T_simulation, label='Firing rate')

        # Produce spikes with the signal
        spike_times_thin = produce_spikes(firing_rate, dt, T_simulation, remove_start)
        spike_train.append(spike_times_thin)


# Save the file
import cPickle
output_filename = "spike_train.cpickle"
output_filename2 = "positions.cpickle"

# Save spike train
f = open(output_filename,"wb")
cPickle.dump(spike_train, f, protocol=2)
f.close()

# Save positions
f = open(output_filename2, "wb")
cPickle.dump(spike_train, f, protocol=2)
f.close()
