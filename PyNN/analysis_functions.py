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
    result = np.sum(kernel * stimuli[delay,...])
    
    return result


def produce_spikes(firing_rate, dt, T_simulation, remove_start):
    """
    Takes as an input a firing rate dependent on time and outputs a spike train generated with a
    non-homogeneous poisson process. In order to do so, uses the thinning method described in Dayan and Abbot 2001

    Parameters
    --------------------
    firing_rate: The firing rate of a neuron
    dt: the time resolution of the firing rate
    T_simulation: The total time of the signal
    remove_start: the firing rate comes with zeros at the beginning, this is the size of those zeros
    """

    r_max = np.max(firing_rate)  

    # Generate the spiking times
    x_exp = np.random.exponential(1000.0 / r_max, size=firing_rate.size) #This does not need to be so long
    spike_times = np.cumsum(x_exp)  # This is the actual spike_times
    take_out_big_times = spike_times < T_simulation # Indexes that correspond to times bigger than the simulation
    take_out_small_times = remove_start < spike_times  # Indexes of the zero entries
    # We take out the non meaningful spikes
    spike_times = spike_times[take_out_big_times * take_out_small_times]

    # Take the indexes of each spike
    spike_indexes = np.floor(spike_times / dt)  # Maybe it is times this
    spike_indexes = spike_indexes.astype(int)

    # Finally we apply the thinning process
    x = np.random.rand(spike_indexes.size)
    # If the ratio is bigger than a random number we keep the spike
    ratio = firing_rate[spike_indexes] / r_max
    spike_times = spike_times[ratio >= x]

    return spike_times


def calculate_tuning(population, population_orientations, orientation_space, simtime):
    """
    Calculates and plots the mean rate of the cells in population as a function 
    of its orientations. The orientation_space is the space from where the orientations where 
    sampled 
    """
    mean_rate = np.zeros(orientation_space.size)
    
    for index, orientation in enumerate(orientation_space):
        mean_rate[index] = population[population_orientations == orientation].mean_spike_count() * (1000.0/simtime)
    
    return mean_rate


def visualize_conductances_and_voltage(segment, neurons):
    """
    Simple functions to visualize the conductances and the voltage for the neurons
    """

    gexc = segment.analogsignalarrays[0]
    ginh = segment.analogsignalarrays[1]
    v = segment.analogsignalarrays[2]

    plt.subplot(1, 2, 1)
    plt.plot(gexc.times, gexc[:, neurons], label='exc')
    plt.plot(ginh.times, ginh[:, neurons], label='inh')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(v.times, v[:,neurons], label='v')

    plt.legend()
    plt.show()


def visualize_conductances(segment, neurons):
    """
    Visualize both the excitatory and inhibitory conductances of the neurons array
    """

    gexc = segment.analogsignalarrays[0]
    ginh = segment.analogsignalarrays[1]

    plt.subplot(2, 1, 1)
    plt.plot(gexc.times, gexc[:,neurons], label='exc')
    plt.xlabel('Time (ms)')
    plt.ylabel('Conductance (uS)')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(ginh.times, ginh[:, neurons], label='inh')
    plt.xlabel('Time (ms)')
    plt.ylabel('Conductance (uS)')
    plt.legend()

    plt.show()


def conductance_analysis(population, segment, orientation_space, population_orientations):
    """
    Plots DC and F1 for the conductance with respect to the orientation of the neurons
    in population
    """

    neuron_with_orientation = np.zeros(orientation_space.size)

    for index, orientation in enumerate(orientation_space):
        aux = np.where(population_orientations == orientation)[0][0]
        neuron_with_orientation[index] = population.id_to_index(population[aux])

    g_exc = segment.analogsignalarrays[0]

    F1 = np.zeros(orientation_space.size)
    DC = np.zeros(orientation_space.size)

    for index, neuron in enumerate(neuron_with_orientation):
        DC[index] = np.mean(g_exc[:, neuron])
        F1[index] = np.max(np.array(g_exc[:, neuron]) - DC[index])

    return DC, F1
