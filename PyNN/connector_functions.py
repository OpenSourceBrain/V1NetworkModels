import numpy as np
from scipy.stats.stats import pearsonr
from kernel_functions import spatial_kernel, gabor_kernel
from misc_functions import circular_dist, normal_function
from math import cos, sin, exp
import cPickle


def normalize_connection_list(connection_list):
    """
    This function takes a list of tuples represnting the connections from one neuron population to another and returns
    the same list with the connection weights (the third element of the list's tuples) normalized/averaged
    """
    weights = [x[2] for x in connection_list] # Make a list with the weights
    average = sum(weights) / len(weights)  # Calculate the average of the weights

    return [(x[0], x[1], average, x[3]) for x in connection_list]

##############################################################
# LGN related functions
##############################################################


def load_positions():

    directory = './data/'
    format = '.cpickle'

    ## Load space
    polarity = '_on'
    mark = 'positions'
    positions_filename = directory + mark + polarity + format
    f1 = open(positions_filename, "rb")
    positions_on = cPickle.load(f1)
    f1.close()

    polarity = '_off'
    mark = 'positions'
    positions_filename = directory + mark + polarity + format
    f1 = open(positions_filename, "rb")
    positions_off = cPickle.load(f1)
    f1.close()

    return positions_on, positions_off


def load_lgn_spikes(contrast, number_of_lgn_layers):
    """
    Returns a list with the spikes associated to the on-center and off-center cells in the lgn respectively
    """

    directory = './data/'
    format = '.cpickle'

    spikes_on = []
    spikes_off = []

    for layer in xrange(number_of_lgn_layers):

        #  Layer 1
        layer = '_layer' + str(layer)

        polarity = '_on'
        contrast_mark = str(contrast)
        mark = '_spike_train'
        spikes_filename = directory + contrast_mark + mark + polarity + layer + format
        f2 = open(spikes_filename, 'rb')
        spikes_on.append(cPickle.load(f2))
        f2.close()

        polarity = '_off'
        contrast_mark = str(contrast)
        mark = '_spike_train'
        spikes_filename = directory + contrast_mark + mark + polarity + layer + format
        f2 = open(spikes_filename, 'rb')
        spikes_off.append(cPickle.load(f2))
        f2.close()

    return spikes_on, spikes_off


def return_lgn_starting_coordinates(lgn_positions, Nside_lgn):
    """
    Given one file with the positions returns x0, y0, dx, dy
    """

    x0, y0 = lgn_positions[0] # Take out the lower corner from the positions
    x_end, y_end = lgn_positions[-1]
    x1, dummy = lgn_positions[Nside_lgn]
    dummy, y1 = lgn_positions[1]
    dx = x1 - x0
    dy = y1 - y0

    return x0, y0, dx, dy


##############################################################
# Thalamo-Cortical connections
##############################################################


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

    transforms_to_radians = np.pi / 180
    theta *= transforms_to_radians
    phi *= transforms_to_radians  # Transforms to radians

    # Translate
    x = x - xc
    y = y - yc

    # Rotate
    aux1 = cos(theta) * x + sin(theta) * y
    y = -sin(theta) * x + cos(theta) * y

    x = aux1

    # Function
    r = x**2 + (gamma * y) ** 2
    exp_part = exp(- r / (2 * sigma**2))
    cos_part = cos(2 * np.pi * w * x + phi)

    return exp_part * cos_part


def create_thalamocortical_connection(source, target, polarity, n_pick, g, delay, sigma, gamma, w, phases, orientations, simulator):
    """
    Creates a connection from a layer in the thalamus to a layer in the cortex through the mechanism of Gabor sampling
    """

    # Produce a list with the connections
    connections_list = create_lgn_to_cortical(source, target, polarity, n_pick, g, delay, sigma, gamma, phases, w, orientations)

    # Normalize connection
    connections_list = normalize_connection_list(connections_list)

    # Transform it into a connector
    connector = simulator.FromListConnector(connections_list, column_names=["weight", "delay"])

    # Create the excitatory and inhibitory projections
    simulator.Projection(source, target, connector, receptor_type='excitatory')
    simulator.Projection(source, target, connector, receptor_type='inhibitory')


def create_lgn_to_cortical(lgn_population, cortical_population, polarity,  n_pick, g, delay,  sigma, gamma, phases,
                           w, orientations):
    """
    Creates the connection from the lgn population to the cortical population with a gabor profile. It also extracts
    the corresponding gabor parameters that are needed in order to determine the connectivity.
    """

    print 'Creating connection from ' + lgn_population.label + ' to ' + cortical_population.label

    # Initialize connections
    connections = []

    for cortical_neuron in cortical_population:
        # Set the parameters
        x_cortical, y_cortical = cortical_neuron.position[0:2]
        cortical_neuron_index = cortical_population.id_to_index(cortical_neuron)
        theta = orientations[cortical_neuron_index]
        phi = phases[cortical_neuron_index]

        # Create the connections from lgn to cortical_neuron
        #lgn_to_cortical_connection(cortical_neuron_index, connections, lgn_population, n_pick, g, polarity, sigma,
        #gamma, phi, w, theta, x_cortical, y_cortical)

        lgn_to_cortical_connection(cortical_neuron_index, connections, lgn_population, n_pick, g, delay, polarity, sigma,
                                   gamma, phi, w, theta, 0, 0)

    return connections


def lgn_to_cortical_connection(cortical_neuron_index, connections, lgn_neurons, n_pick, g, delay, polarity, sigma,
                               gamma, phi, w, theta, x_cortical, y_cortical):
    """
    Creates connections from the LGN to the cortex with a Gabor profile.

    This function adds all the connections from the LGN to the cortical cell with index = cortical_neuron_index. It
    requires as parameters the cortical_neruon_index, the current list of connections, the lgn population and also
    the parameters of the Gabor function.

    Parameters
    ----
    cortical_neuron_index : the neuron in the cortex -target- that we are going to connect to
    connections: the list with the connections to which we will append the new connnections
    lgn_neurons: the source population
    n_pick: How many times we will sample per neuron
    g: how strong is the connection per neuron
    delay: the time it takes for the action potential to arrive to the target neuron from the source neuron
    polarity: Whether we are connection from on cells (polarity = 1) or off cells (polarity = -1)
    sigma: Controls the decay of the exponential term
    gamma: x:y proportionality factor, elongates the pattern
    phi: Phase of the overall pattern
    w: Frequency of the pattern
    theta: Rotates the whole pattern by the angle theta
    x_cortical, y_cortical : The spatial coordinate of the cortical neuron

    """

    for lgn_neuron in lgn_neurons:
            # Extract position
            x, y = lgn_neuron.position[0:2]
            # Calculate the gabor probability
            probability = polarity * gabor_probability(x, y, sigma, gamma, phi, w, theta, x_cortical, y_cortical)
            probability = np.sum(np.random.rand(n_pick) < probability)  # Samples

            synaptic_weight = (g / n_pick) * probability
            lgn_neuron_index = lgn_neurons.id_to_index(lgn_neuron)

            # The format of the connector list should be pre_neuron, post_neuron, w, tau_delay
            if synaptic_weight > 0:
                connections.append((lgn_neuron_index, cortical_neuron_index, synaptic_weight, delay))



#######################################################################
# Intra-cortical connections phase and orientation distance version
#######################################################################

def create_cortical_to_cortical_connection(source_population, target_population, source_orientations, source_phases,
                                           target_orientations, target_phases, orientation_sigma, phase_sigma,
                                           g, delay,  n_pick, target_type_excitatory=True):
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
        cortical_to_cortical_connection(target_neuron_index, connections, source_population, n_pick, g, delay,
                                        source_orientations, source_phases, orientation_sigma, phase_sigma,
                                        target_neuron_orientation, target_neuron_phase, target_type=target_type_excitatory)

    return connections


def cortical_to_cortical_connection(target_neuron_index, connections, source_population, n_pick, g, delay, source_orientations,
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
        or_distance = circular_dist(target_neuron_orientation, source_neuron_orientation, 180)

        if target_type:
            phase_distance = circular_dist(target_neuron_phase, source_neuron_phase, 360)
        else:
            phase_distance = 180 - circular_dist(target_neuron_phase, source_neuron_phase, 360)

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
                    connections.append((source_neuron_index, target_neuron_index, synaptic_weight, delay))

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

    # Now we calculate the cross-correlation of this cell with each other cell in the gri
    for x_to in x_values:
        for y_to in y_values:

            # Call the receptive field of the cell
            Z2 = spatial_kernel(lx, dx, ly, dy, sigma_center, sigma_surround, inverse=1, x_tra=x_to, y_tra=y_to)

            # Calculate the correlation and sotre it
            values[counter_aux] = pearsonr(Z1.flat, Z2.flat)[0]
            counter_aux += 1
    return values


##############################################################
# Intra-cortical connections Gabor Correlation
##############################################################


def create_cortical_to_cortical_connection_corr(source_population, target_population, source_orientations, source_phases,
                                           target_orientations, target_phases, orientation_sigma, phase_sigma, g, delay,
                                           n_pick, target_type_excitatory=True):
    """
    Creates the connections from source population to target population in the cortex.
    """
    print 'Creating connection from ' + source_population.label + ' to ' + target_population.label
    connections = []

    # Gabor parameters
    w = 0.8
    gamma = 1  # Aspect ratio
    sigma = 1
    # Space parameters
    dx = 0.1
    lx = 6.0
    dy = 0.1
    ly = 6.0

    for target_neuron in target_population:

        # Extract target
        x_target, y_target = target_neuron.position[0:2]
        target_neuron_index = target_population.id_to_index(target_neuron)
        target_neuron_orientation = target_orientations[target_neuron_index]
        target_neuron_phase = target_phases[target_neuron_index]

        #Z1 = gabor_kernel(lx, dx, ly, dy, sigma, gamma, target_neuron_phase, w,
        #                  target_neuron_orientation, x_target, y_target)

        Z1 = gabor_kernel(lx, dx, ly, dy, sigma, gamma, target_neuron_phase, w,
                          target_neuron_orientation, 0, 0 )

        # Create the connection from source to target_neuron
        cortical_to_cortical_connection_corr(target_neuron_index, connections, source_population, n_pick, g, delay,
                                        source_orientations, source_phases, target_neuron_orientation,
                                        target_neuron_phase, Z1, lx, dx, ly, dy, sigma, gamma, w,
                                        target_type=target_type_excitatory)

    return connections


def cortical_to_cortical_connection_corr(target_neuron_index, connections, source_population, n_pick, g, delay,
                                    source_orientations, source_phases, target_neuron_orientation, target_neuron_phase,
                                    Z1, lx, dx, ly, dy, sigma, gamma, w, target_type):
    """
    Creates the connections from the source population to the target neuron

    """
    for source_neuron in source_population:
        # Extract index, orientation and phase of the target
        x_source, y_source = source_neuron.position[0:2]
        source_neuron_index = source_population.id_to_index(source_neuron)
        source_neuron_orientation = source_orientations[source_neuron_index]
        source_neuron_phase = source_phases[source_neuron_index]

        #Z2 = gabor_kernel(lx, dx, ly, dy, sigma, gamma, source_neuron_phase, w, source_neuron_orientation,
        #                  x_source, y_source)

        Z2 = gabor_kernel(lx, dx, ly, dy, sigma, gamma, source_neuron_phase, w, source_neuron_orientation,
                          0, 0)

        if target_type:
            probability = pearsonr(Z1.flat, Z2.flat)[0]
        else:
            probability = (-1) * pearsonr(Z1.flat, Z2.flat)[0]

        probability = np.sum(np.random.rand(n_pick) < probability)  # Samples
        synaptic_weight = (g / n_pick) * probability

        if synaptic_weight > 0:
                    connections.append((source_neuron_index, target_neuron_index, synaptic_weight, delay))

    return connections

