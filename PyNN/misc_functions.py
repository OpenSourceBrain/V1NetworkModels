"""
This file contains miscellaneous functions 
"""

import numpy as np 
from math import sqrt

def create_standar_indexes(dt, dt_kernel, dt_stimuli, kernel_size, Nt_simulation):
    """
    Create the signal, delay and stimuli indexes
    """

    # Scale factors (multiply to transform from one set of coordinates to the other)
    input_to_image = dt / dt_stimuli # Transforms input to image
    kernel_to_input = dt_kernel / dt  # Transforms kernel to input 
    input_to_kernel = dt / dt_kernel #  Transforms input to kernel
    
    # Here we remove the indexes of the signal that do not 
    # have a corresponding image and then we got its indexes 
    remove_start = int(kernel_size * kernel_to_input)
    signal_indexes = np.arange(remove_start, Nt_simulation).astype(int)
        
    # Delay indexes 
    delay_indexes = np.floor(np.arange(kernel_size) * kernel_to_input)
    delay_indexes = delay_indexes.astype(int)   
    
    # Stimuli indexes 
    stimuli_indexes = np.floor(np.arange(Nt_simulation) * input_to_image)
    stimuli_indexes = stimuli_indexes.astype(int)
    
    return signal_indexes, delay_indexes, stimuli_indexes

def create_extra_indexes(kernel_size, Nt_simulation): 
    """
    Create the working indexes and the kernel times 
    """
    
    # Index of the times units 
    working_indexes = np.arange(Nt_simulation).astype(int) 
    
    # Calculate kernel times 
    kernel_times = np.arange(kernel_size)
    kernel_times = kernel_times.astype(int) 
    
    return working_indexes, kernel_times



def normal_function(x, mean=0, sigma=1.0):
    """
    Returns the value of probability density of normal distribution N(mean,sigma) at point `x`.
    """
    _normalization_factor = sqrt(2 * np.pi)

    return np.exp(-np.power((x - mean)/sigma, 2)/2) / (sigma * _normalization_factor)


def circular_dist(a, b, period):
    """
    Returns the distance between a and b (scalars) in a domain with `period` period.
    """
    return np.minimum(np.abs(a - b), period - np.abs(a - b))

