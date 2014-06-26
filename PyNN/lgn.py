from pyNN.utility import get_script_args, Timer
import numpy as np
import matplotlib.pyplot as plt 
import pyNN.space as space
from connector_functions import gabor_probability

#simulator_name = get_script_args(1)[0]
#exec("import pyNN.%s as simulator" % simulator_name)
import pyNN.nest as simulator

N = 30 * 30
N = 3 * 3  # Number of neurons
t = 150.0  # Simulation time

# Has to be called at the beginning of the simulation
simulator.setup(timestep=0.1, min_delay=0.1, max_delay=10)

def spike_times(i):
    '''
    Test function
    '''

    A = []
    for k in range(len(i)):
        A.append(np.arange(1,k,0.1))

    return A


# Spatial structure
lgn_structure = space.Grid2D(aspect_ratio=1, x0=-6.0, y0=-6.0, dx=0.1, dy=0.1, z=0)
# Neuron Model
model = simulator.SpikeSourceArray(spike_times=spike_times)

# Populations
lgn_neurons = simulator.Population(N, model, structure=lgn_structure, label='LGN')
retinal_neurons = simulator.Population(1, simulator.IF_curr_exp())



simulator.run(t)  # Run the simulations for t ms
simulator.end()

connector = []

example_neuron = lgn_neurons[3]

w = 0.8
phi = 0 * np.pi
gamma = 1
sigma = 1
theta = 0.0 * np.pi

g_exc = 0.98
n_pick = 3

for neuron in lgn_neurons:
    x = neuron.position[0]
    y = neuron.position[1]
    probability = gabor_probability(x, y, sigma, gamma, phi, w, theta)
    probability = np.sum(np.random.rand(n_pick) < probability)
    print 'probability', probability
    synaptic_weight = (g_exc / n_pick) * probability
    neuron_index = lgn_neurons.id_to_index(neuron)
    print 'neuron index', neuron_index
    print 'x=', neuron.position[0]
    print 'y=', neuron.position[1]

    connector.append((neuron_index, 0, synaptic_weight, 0.1))




