"""
This function creates and stores the LGN spikes patterns. It creates and stores them in an order that is consistent 
with the one that PyNN uses to create the spike arrays. That is, for each x it will first sweep completely the y coordaintes
and the it will change x 
"""

import numpy as np
import cPickle
import matplotlib.pyplot as plt 

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
T_simulation = 10 * 10 ** 3.0  # ms
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
Phi = 0 * np.pi # Phase of the pattern 
Theta = 0 * np.pi # Orientation of the pattern
max_contrast = 2.4 * 2 
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

# Create the stimuli
stimuli = sine_grating(dx, lx, dy, ly, A, K, Phi, Theta, dt_stimuli, N_stimuli, w)

# Create the cell array
Ncells = 30
lx_cells = 5.0
ly_cells = 5.0
on_off_cells = 1 # 1 and -1 for on and off cells respectively 

half_lattice_x = 0.5 * lx_cells / Ncells
half_lattice_y = 0.5 * ly_cells / Ncells

# Coordinates of the cells to be simulated 
x_values = np.linspace(-lx_cells/2 , lx_cells/2, Ncells, endpoint=True) 
y_values = np.linspace(-ly_cells/2, ly_cells/2, Ncells, endpoint=True)

number_of_layers = 4 

for layer in xrange(number_of_layers):

    spike_train = []
    positions =[]
    counter = 0
    
    number_of_spikes_array = np.zeros((Ncells, Ncells))
    
    counter1 = 0 
    counter2 = 0
    
    for x in x_values:
        for y in y_values:
            print '----------'
            print 'Creating spikes for layer number ' + str(layer)
            print 'Order of creation in the grid'
            print 'x=', x
            print 'y=', y
            print 'Steps to finish', Ncells * Ncells - counter
            counter += 1 
    
            xc = x
            yc = y
    
            positions.append((x,y))
    
    
            # Create the kernel
            kernel = create_kernel(dx, lx, dy, ly, sigma_surround, sigma_center,
                                   dt_kernel, kernel_size, inverse=on_off_cells, x_tra=xc, y_tra=yc)
            
            # Initialize the signal array to be filled 
            firing_rate = np.zeros(Nt_simulation)
    
            # Calculate the firing rate
            for index in signal_indexes:
                firing_rate[index] = convolution(index, kernel_times, delay_indexes, stimuli_indexes, kernel, stimuli)
    
            firing_rate += 10  # Add background noise for on cells
            #firing_rate += 15  # Add background noise for off cells
            
            # Rectify the firing rate
            firing_rate[ firing_rate < 0] = 0
                    
            # Produce spikes with the signal
            spike_times_thin = produce_spikes(firing_rate, dt, T_simulation, remove_start)
            spike_times_thin -= remove_start # Translate the spikes to the origin  
            spike_train.append(spike_times_thin)
            
            # Count the spikes in the cell 
            number_of_spikes_array[counter1, counter2] = len(spike_times_thin)
            print 'counters 1, 2=', counter1, counter2
            counter1 += 1
            
        counter2 += 1 
        counter1 = 0
    
    # Save the file
    folder = './data/'
    format = '.cpickle'
    output_filename = folder + 'spike_train_on_layer' + str(layer) + format
    output_filename2 = folder +  'positions_on_layer' + str(layer) + format
    
    # Save spike train
    f = open(output_filename,"wb")
    cPickle.dump(spike_train, f, protocol=2)
    f.close()
    
    # Save positions
    f = open(output_filename2, "wb")
    cPickle.dump(positions, f, protocol=2)
    f.close()
    
    # Show how many spikes per position 
    plt.subplot(2, 2, layer)
    plt.imshow(number_of_spikes_array, interpolation='None', extent=[-lx_cells/2,lx_cells/2,ly_cells/2,-ly_cells/2])
    plt.colorbar()


# At the end show the maps of firing rates 
plt.show()