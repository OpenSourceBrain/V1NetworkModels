import matplotlib.pyplot as plt
import numpy as np


def sta_v(signal, ims, training_indexes, delay_indexes, image_indexes, kernel_times, verbose=True):
    '''
    Calculate the voltage triggered average of a signal with respected to a group of images ims.

    Parameters
    ------------------------------
    training_indexes: Should be the indexes of the actual part of the signal where the average is
    going to be calculated.
     
    delay_indexes: Correspond to the indexes of the signal equivalent to the delays in the kernel. 
    
    image_indexes: A vector that matches indexes in the signal to corresponing indexes in the 
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
        delay = image_indexes[training_indexes - delay_index] # Take the image indexes
        weighted_stimuli = np.sum( signal[training_indexes, np.newaxis, np.newaxis] * ims[delay,...], axis=0)
        sta[tau,...] = weighted_stimuli / training_indexes.size

    #return np.transpose(sta, (0,2,1))
    return sta


def convolution(index, kernel_times, delay_indexes, image_indexes, kernel, stimuli):
    '''
    Calculates the convolution between the kernel and the image. 
    Parameters:

    '''
    # Calculate proper delays in the image indexes 
    delay = image_indexes[index - delay_indexes] 
    # Do the calculation    
    result = np.sum(kernel[kernel_times,...] * stimuli[delay,...])
    
    return result


