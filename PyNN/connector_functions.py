import numpy as np
from scipy.stats.stats import pearsonr
from kernel_functions import spatial_kernel
from misc_functions import circular_dist, normal_function

def gabor_probability(x, y, sigma, gamma, phi, w, theta, xc=0, yc=0):

    """
    calculate the gabor function of x and y

    Returns value of the 2D Gabor function at x, y

    sigma: Controls the decay of the exponential term
    gamma: x:y proportionality factor, elongates the pattern
    phi: Phase of the overall pattern
    w: Frequency of the pattern
    theta: Rotates the whole pattern by the angle theta
    xc, yc : Linear translation
    """

    # Translate
    x -= xc
    y -= yc

    # Rotate
    x = np.cos(theta) * x + np.sin(theta) * y
    y = -np.sin(theta) * x + np.cos(theta) * y

    # Function
    exp_part = np.exp(-(x**2 + (gamma * y)**2)/(2 * sigma**2))
    cos_part = np.cos(2 * np.pi * w * x + phi)

    return exp_part * cos_part

def lgn_to_cortical_connection(cortical_neuron_index, connections, lgn_neurons, n_pick, g_exc, polarity, sigma,
                               gamma, phi, w, theta, x_cortical, y_cortical):
    """
    Creates connections from the LGN to the cortex with a Gabor profile.

    This function adds all the connections from the LGN to the cortical cell with index = cortical_neuron_index. It
    requieres as parameters the cortical_neruon_index, the current list of connections, the lgn population and also
    the parameters of the Gabor function.

     Parameters
     ----

    """

    for lgn_neuron in lgn_neurons:
            # Extract position
            x, y = lgn_neuron.position[0:2]
            # Calculate the gabbor probability
            probability = polarity * gabor_probability(x, y, sigma, gamma, phi, w, theta, x_cortical, y_cortical)
            probability = np.sum(np.random.rand(n_pick) < probability)  # Samples

            synaptic_weight = (g_exc / n_pick) * probability
            lgn_neuron_index = lgn_neurons.id_to_index(lgn_neuron)

            # The format of the connector list should be pre_neuron, post_neuron, w, tau_delay
            if synaptic_weight > 0:
                connections.append((lgn_neuron_index, cortical_neuron_index, synaptic_weight, 0.1))


def create_lgn_to_cortical(lgn_population, cortical_population, polarity,  n_pick, g_exc, sigma, gamma, phases,
                           w, orientations):
    """
    Creates the connection from the lgn population to the cortical population with a gabor profile
    """

    print 'Creating connection from ' + lgn_population.label + ' to ' + cortical_population.label

    # Intiliaze connection
    connections = []

    for cortical_neuron in cortical_population:
        # Set the parameters
        x_cortical, y_cortical = cortical_neuron.position[0:2]
        cortical_neuron_index = cortical_population.id_to_index(cortical_neuron)
        theta = orientations[cortical_neuron_index]
        phi = phases[cortical_neuron_index]

        # Create the connections from lgn to cortical_neuron
        lgn_to_cortical_connection(cortical_neuron_index, connections, lgn_population, n_pick, g_exc, polarity, sigma,
                                   gamma, phi, w, theta, x_cortical, y_cortical)

    return connections


def cortical_to_cortical_connection(target_neuron_index, connections, source_population, n_pick, g, source_orientations,
                                    source_phases, orientation_sigma, phase_sigma, target_neuron_orientation,
                                    target_neuron_phase, target_type):
    """
    Creates the connections from the source population to the target neuron

    """
    for source_neuron in source_population:
        # Extract index, orientation and phase of the target
        source_neuron_index = source_population.id_to_index(source_neuron)
        source_neuron_orientation = source_orientations[source_neuron_index]
        source_neuron_phase = source_phases[source_neuron_index]

        # Now calculate phase and orientation distances
        or_distance = circular_dist(target_neuron_orientation, source_neuron_orientation, np.pi)

        if target_type:
            phase_distance = circular_dist(target_neuron_phase, source_neuron_phase, 2*np.pi)
        else:
            phase_distance = np.pi - circular_dist(target_neuron_phase, source_neuron_phase, 2*np.pi)

        # Now calculate the gaussian function
        or_gauss = normal_function(or_distance, mean=0, sigma=orientation_sigma)
        phase_gauss = normal_function(phase_distance, mean=0, sigma=phase_sigma)

        # Now normalize by guassian in zero
        or_gauss = or_gauss / normal_function(0, mean=0, sigma=orientation_sigma)
        phase_gauss = phase_gauss / normal_function(0, mean=0, sigma=phase_sigma)

        # Probability is the product
        probability = or_gauss * phase_gauss
        probability = np.sum(np.random.rand(n_pick) < probability)  # Samples
        synaptic_weight = (g / n_pick) * probability

        if synaptic_weight > 0:
                    connections.append((source_neuron_index, target_neuron_index, synaptic_weight, 0.1))


def create_cortical_to_cortical_connection(source_population, target_population, source_orientations, source_phases,
                                           target_orientations, target_phases, orientation_sigma, phase_sigma,
                                           g, n_pick, target_type_excitatory=True):
    """
    Creates the connections from source population to target population in the cortex.
    """
    print 'Creating connection from ' + source_population.label + ' to ' + target_population.label
    connections = []

    for target_neuron in target_population:

        # Extract targe parameters
        target_neuron_index = target_population.id_to_index(target_neuron)
        target_neuron_orientation = target_orientations[target_neuron_index]
        target_neuron_phase = target_phases[target_neuron_index]

        # Create the connection from source to target_neuron
        cortical_to_cortical_connection(target_neuron_index, connections, source_population, n_pick, g,
                                        source_orientations, source_phases, orientation_sigma, phase_sigma,
                                        target_neuron_orientation, target_neuron_phase, target_type=target_type_excitatory)


    return connections

def calculate_correlations_to_cell(x_position, y_position, x_values, y_values,
                                   lx, dx, ly, dy, sigma_center, sigma_surround):
    """
    Calculates the correlations of the cell in x_positions, y_position to all other cells in the population

    Returns a vector with the pearson correlation coefficient between the cell and all other cells

    Parameters
    ---------------
    x_position, y_position : The cell's spatial coordinates
    x_values, y_values : The grid coordinates where all the other cells are located
    lx, ly : The extent of the receptive field space in the x and y direction respectively
    dx, dy : The resolution of the receptive field space in x and y respectively
    sigma_center, sigma_surround: The parameter of the center-surround receptive field structure


    """
    values = np.zeros(x_values.size * y_values.size)
    counter_aux = 0

    # Calculate the receptive field of the cell
    Z1 = spatial_kernel(lx, dx, ly, dy, sigma_center, sigma_surround, inverse=1, x_tra=x_position, y_tra=y_position)

    # Now we calclate the cross-correlation of this cell with each other cell in the gri
    for x_to in x_values:
        for y_to in y_values:

            # Call the receptive field of the cell
            Z2 = spatial_kernel(lx, dx, ly, dy, sigma_center, sigma_surround, inverse=1, x_tra=x_to, y_tra=y_to)

            # Calculate the correlation and sotre it
            values[counter_aux] = pearsonr(Z1.flat, Z2.flat)[0]
            counter_aux += 1
    return values