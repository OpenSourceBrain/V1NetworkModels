import matplotlib.pyplot as plt
import numpy as np
from kernel_functions import create_kernel


#Time parameters  
dt = 1.0  #milliseconds
dt_kernel = 5.0 #milliseconds
dt_stimuli = 10.0  # milliseconds

kernel_size = 25 # The size of the kernel 

T_simulation = 8 * 10 ** 4.0 # Total time of the simulation in ms
Nt_simulation = int(T_simulation / dt) # Number of simulation points 
N_stimuli = int(T_simulation / dt_stimuli) #Number of stimuli

# Space parameters 
dx = 1.0
Lx = 20.0
Nx = int(Lx / dx)
dy = 1.0
Ly = 20.0
Ny = int(Ly / dy )


# Call the kernel 
sigma_center = 15  # Size of center area in the center-surround profile 
sigma_surround = 3  # Size of surround area in the center-surround profile 
kernel = create_kernel(dx, Lx, dy, Ly, sigma_surround, sigma_center, dt_kernel, kernel_size) 

# Call the stimuli 
stimuli = np.random.randint(-1, 2, size=(N_stimuli, Nx, Ny))

# Initialize the signal
signal = np.zeros(Nt_simulation)

# Scale factors 
input_to_image = dt / dt_stimuli  # Transforms input to image
kernel_to_input = dt_kernel / dt  # Transforms kernel to input 
input_to_kernel = dt / dt_kernel  # Transforms input to kernel   

working_indexes = np.arange(Nt_simulation).astype(int) # From here we remove the start at put the ones
remove_start = int(kernel_size * kernel_to_input)
signal_indexes = np.arange(remove_start, Nt_simulation).astype(int)

# Calculate kernel
kernel_times = np.arange(kernel_size)
kernel_times = kernel_times.astype(int) # Make the values indexes 
    
# Delay indexes 
delay_indexes = np.floor(kernel_times * kernel_to_input)
delay_indexes = delay_indexes.astype(int) # Make the values indexes 

# Image Indexes 
stimuli_indexes = np.zeros(working_indexes.size)
stimuli_indexes[working_indexes] = np.floor(working_indexes * input_to_image)
stimuli_indexes = stimuli_indexes.astype(int)

for index in signal_indexes:
    delay = stimuli_indexes[index - delay_indexes] 
    # Do the calculation    
    signal[index] = np.sum(kernel[kernel_times,...] * stimuli[delay,...])

t = np.arange(remove_start*dt, T_simulation, dt)
plt.plot(t, signal[signal_indexes], '-', label='Kernel convoluted with noise')
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Convolution')
plt.grid()
plt.show()


## Calculate the STA
kernel_size = kernel_times.size
Nside = np.shape(stimuli)[2]
sta = np.zeros((kernel_size ,Nside, Nside))

for tau, delay_index in zip(kernel_times, delay_indexes):
    # For every tau we calculate the possible delay and take the appropriate image index
    delay = stimuli_indexes[signal_indexes - delay_index] 
    # Now we multiply the voltage for the appropriate images 
    weighted_stimuli = np.sum( signal[signal_indexes, np.newaxis, np.newaxis] * stimuli[delay,...], axis=0)
    # Finally we divide for the sample size 
    sta[tau,...] = weighted_stimuli / signal_indexes.size


## Visualize the STA 
closest_square_to_kernel = int(np.sqrt(kernel_size)) ** 2

# Define the color map
cdict1 = {'red':   ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.1),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 1.0),
                   (0.5, 0.1, 0.0),
                   (1.0, 0.0, 0.0))
        }

from matplotlib.colors import LinearSegmentedColormap
blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1)

n = int( np.sqrt(closest_square_to_kernel))
# Plot the filters 
for i in range(closest_square_to_kernel):
    plt.subplot(n,n,i + 1)
    plt.imshow(sta[i,:,:], interpolation='bilinear', cmap=blue_red1)
    plt.colorbar()

plt.show()
