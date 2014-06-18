import numpy as np
import matplotlib.pyplot as plt 
from kernel_functions import create_kernel
from analysis_functions import convolution
from stimuli_functions import ternary_noise, sine_grating 

#Time parameters  
dt = 0.25  #milliseconds
dt_kernel = 1.0 #milliseconds
dt_stimuli = 10.0  # milliseconds
kernel_size = 25

# Space parameters 
dx = 1.0
Lx = 20.0
Nx = (Lx / dx)
dy = 1.0
Ly = 20.0
Ny = int(Ly / dy )


sigma_center = 15
sigma_surround = 3

# sine gratin spatial parameters 
K = 1
Phi = 1 * np.pi
Theta = 0 * np.pi
A = 1.0
# Temporal frequency of sine grating 
w = 0.005


# Set the random set for reproducibility
seed = 1053
np.random.seed(seed)

T_simulation = 2 * 10 ** 3.0 # ms 
Nt_simulation = int(T_simulation / dt)  # Number of simulation points 
N_stimuli = int(T_simulation / dt_stimuli)  # Number of stimuli points 

# Scale factors (multiply to transform from one set of coordinates to the other)
input_to_image = dt / dt_stimuli # Transforms input to image
kernel_to_input = dt_kernel / dt  # Transforms kernel to input 
input_to_kernel = dt / dt_kernel #  Transforms input to kernel

# Index of the times units 
working_indexes = np.arange(Nt_simulation).astype(int) 
# Here we remove the indexes of the signal that do not 
# have a corresponding image and then we got its indexes 
remove_start = int(kernel_size * kernel_to_input)
signal_indexes = np.arange(remove_start, Nt_simulation).astype(int)

# Calculate kernel
kernel_times = np.arange(kernel_size)
kernel_times = kernel_times.astype(int) 
    
# Delay indexes 
delay_indexes = np.floor(kernel_times * kernel_to_input)
delay_indexes = delay_indexes.astype(int)   

# Image Indexes 
stimuli_indexes = np.zeros(working_indexes.size)
stimuli_indexes[working_indexes] = np.floor(working_indexes * input_to_image)
stimuli_indexes = stimuli_indexes.astype(int)

# Initialize the signal array to be filled 
signal = np.zeros(Nt_simulation)

kernel = create_kernel(dx, Lx, dy, Ly, sigma_surround, sigma_center, dt_kernel, kernel_size)
# stimuli = ternary_noise(N_stimuli, Nx, Ny)
stimuli = sine_grating(dx, Lx, dy, Ly, A, K, Phi, Theta, dt_stimuli, N_stimuli, w)

for index in signal_indexes:
    signal[index] = convolution(index, kernel_times, delay_indexes, stimuli_indexes, kernel, stimuli)

# Rectify the signal
signal[ signal < 0 ] = 0 

# Visualize the signal 
t = np.arange(remove_start*dt, T_simulation, dt)
plt.plot(t, signal[signal_indexes], '-', label='signal')
plt.hold(True)

rmax = np.max(signal) / 1000
signal_size = signal.size

# Generate the spiking times 
x_exp = np.random.exponential(1.0 / rmax, size=signal_size) #This does not to be so long 
spike_times =  np.cumsum(x_exp) # This is the actual spike_times 
spike_times  = spike_times[ spike_times <= T_simulation] # This should not be necessary 
# Now we need to transform spike times in indexes of the signal

spike_indexes = np.floor(spike_times / dt )
spike_indexes = spike_indexes.astype(int)

# Thin them 
x = np.random.rand(spike_indexes.size)
ratio = signal[spike_indexes] / ( 1000 * rmax)
index_to_keep = np.where( ratio  >=  x ) #Thin
spike_times_thin = spike_times[index_to_keep]

plt.plot(spike_times, x * np.max(signal), label='xrand')

y = np.ones_like(spike_times) * 10
plt.plot(spike_times, y, '*', label='rmax')

y2 = np.ones_like(spike_times_thin) * 2
plt.plot(spike_times_thin, y2, '*', label='thined')

plt.legend()
plt.show()
