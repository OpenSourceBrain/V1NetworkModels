import matplotlib.pyplot as plt
import numpy as np
from math import cos, sin 

def ternary_noise(N_stimuli, Nx, Ny):
    '''
    Returns ternary noise with values -1, 0 and 1 
    '''
    return  np.random.randint(-1, 2, size=(N_stimuli, Nx, Ny)) 

def sine_grating(dx, Lx, dy, Ly, A, K, Phi, Theta, dt_stimuli, N_stimuli, w):
    '''
    Returns a sine grating stimuli 
    '''
    Nx = int(Lx / dx)
    Ny = int(Ly / dy)
    
    # Transform to appropriate units 
    K = K * 2 * np.pi # Transforms K to cycles per degree
    w = w / 1000.0 # Transforms w to kHz  
    
    x = np.arange(-Lx/2, Lx/2, dx)
    y = np.arange(-Ly/2, Ly/2, dy)
    X, Y = np.meshgrid(x, y)
    Z = A * np.cos(K * X *cos(Theta) + K * Y * sin(Theta) - Phi)
    t = np.arange(0, N_stimuli * dt_stimuli, dt_stimuli)
    f_t = np.cos(w * 2 * np.pi *  t )
    
    stimuli = np.zeros((N_stimuli, Nx, Ny)) 
    
    for k, time_component in enumerate(f_t):
        stimuli[k,...] = Z * time_component
    
    return stimuli