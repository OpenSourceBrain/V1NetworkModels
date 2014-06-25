import matplotlib.pyplot as plt
import numpy as np


def sta_v(signal, ims, training_indexes, delay_indexes, stimuli_indexes, kernel_times, verbose=True):
    '''
    Calculate the voltage triggered average of a signal with respected to a group of images ims.

    Parameters
    ------------------------------
    training_indexes: Should be the indexes of the actual part of the signal where the average is
    going to be calculated.
     
    delay_indexes: Correspond to the indexes of the signal equivalent to the delays in the kernel. 
    
    stimuli_indexes: A vector that matches indexes in the signal to corresponing indexes in the 
    imgages
    
    kernel_times: 
    
    Verbose: Information concerning the calculation  
    '''
    
    kernel_size = kernel_times.size
    Nside = np.shape(ims)[2]
    sta = np.zeros((kernel_size ,Nside, Nside))

    for tau, delay_index in zip(kernel_times, delay_indexes):
        if (verbose == True): # Print the delays
            print 'tau=', tau
        delay = stimuli_indexes[training_indexes - delay_index] # Take the image indexes
        weighted_stimuli = np.sum( signal[training_indexes, np.newaxis, np.newaxis] * ims[delay,...], axis=0)
        sta[tau,...] = weighted_stimuli / training_indexes.size

    #return np.transpose(sta, (0,2,1))
    return sta


def convolution(index, kernel_times, delay_indexes, stimuli_indexes, kernel, stimuli):
    '''
    Calculates the convolution between the kernel and the image. 
    Parameters:

    '''
    # Calculate proper delays in the image indexes 
    delay = stimuli_indexes[index - delay_indexes] 
    # Do the calculation    
    result = np.sum(kernel[kernel_times,...] * stimuli[delay,...])
    
    return result

def produce_spikes(firing_rate, dt, T_simulation):
    """
    Produces spikes from a firing rate with the mechanism of thinning 
    """

    rmax = np.max(firing_rate) / 1000 # Transfors to Hz
    firing_rate_size = firing_rate.size
    
    # Generate the spiking times 
    x_exp = np.random.exponential(1.0 / rmax, size=firing_rate_size) #This does not to be so long 
    spike_times =  np.cumsum(x_exp) # This is the actual spike_times 
    spike_times  = spike_times[ spike_times <= T_simulation] # Take only the ones that are smaller than the simulation time 
    
    # Now we need to transform spike times in indexes of the firing_rate
    spike_indexes = np.floor(spike_times / dt )
    spike_indexes = spike_indexes.astype(int)
    
    # Finally we appliy the thinning process 
    x = np.random.rand(spike_indexes.size)
    ratio = firing_rate[spike_indexes] / ( 1000 * rmax)
    index_to_keep = np.where( ratio  >=  x ) #Thin
    spike_times_thin = spike_times[index_to_keep]

    return spike_times_thin

