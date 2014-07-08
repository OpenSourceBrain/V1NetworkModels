from pyNN.utility import get_script_args, Timer
import numpy as np
import matplotlib.pyplot as plt 
import pyNN.space as space
from connector_functions import gabor_probability
import cPickle

#simulator_name = get_script_args(1)[0]
#exec("import pyNN.%s as simulator" % simulator_name)
import pyNN.nest as simulator

## Load LGN spikes and positions 
format = '.cpickle'
positions_filename = 'positions' + format
f1 = open(positions_filename,"rb")
positions = cPickle.load(f1)
f1.close()

spikes_filename = 'spike_train' + format 
f2 = open(spikes_filename,"rb")
spikes = cPickle.load(f2)
f2.close()

## Initialize network 

Ncells_lgn = 30
Ncell_exc = 40
Ncell_inh = 20

t = 1000.0  # Simulation time

# Has to be called at the beginning of the simulation
simulator.setup(timestep=0.1, min_delay=0.1, max_delay=5.0)

def spike_times(i):
    '''
    Test function
    '''

    A = []
    for k in range(len(i)):
        A.append(spikes[k])

    return A

def spike_times2(i):
    return spikes


# Spatial structure
x0, y0 = positions[0] # Take out the lower corner from the positions 
x_end, y_end = positions[-1]
x1, dummy = positions[Ncells_lgn]
dummy, y1 = positions[1]
dx = x1 - x0
dy = y1 - y0
lx = x_end - x0
ly = y_end - y0

lgn_structure = space.Grid2D(aspect_ratio=1, x0=x0, y0=y0, dx=dx, dy=dy, z=0)
# Neuron Model
lgn_spikes_model = simulator.SpikeSourceArray(spike_times=spike_times2)

# Populations
lgn_neurons = simulator.Population(Ncells_lgn **2, lgn_spikes_model, structure=lgn_structure, label='LGN')
#cortical_neurons_exc = simulator.Population(10, simulator.IF_cond_exp())
cortical_neurons_exc = simulator.Population(Ncell_exc**2, simulator.IF_cond_exp())
#cortical_neurons_inh = simulator.Population(Ncell_inh**2, simulator.IF_cond_exp())

## Create the connector 
connections = []

for cortical_neuron in cortical_neurons_exc:

    w = 0.8  # Spatial frequency
    phi = 0 * np.pi  # Phase
    gamma = 1  # Aspect ratio
    sigma = 1  # Decay ratio
    random_orientation = np.random.rand()
    theta = random_orientation * np.pi # Orientation

    g_exc = 0.98
    n_pick = 3

    connect_to_neuron = cortical_neurons_exc.id_to_index(cortical_neuron) # Connects to this neuron
    print 'connect to neuron', connect_to_neuron

    for lgn_neuron in lgn_neurons:
        x, y = lgn_neuron.position[0:2]
        probability = gabor_probability(x, y, sigma, gamma, phi, w, theta)  # Calculate the gabbor probability
        probability = np.sum(np.random.rand(n_pick) < probability) # Samples
        synaptic_weight = (g_exc / n_pick) * probability
        neuron_index = lgn_neurons.id_to_index(lgn_neuron)

        # The format of the connector list should be pre_neuron, post_neuron, w, tau_delay
        connections.append((neuron_index, connect_to_neuron, synaptic_weight, 0.1))

# Make the list a connector
connector = simulator.FromListConnector(connections, column_names=["weight", "delay"])

excitatory_connections = simulator.Projection(lgn_neurons, cortical_neurons_exc, connector,
                                              simulator.StaticSynapse())


lgn_neurons.record('spikes')
cortical_neurons_exc.record('gsyn_exc')

simulator.run(t)  # Run the simulations for t ms
simulator.end()

data = lgn_neurons.get_data()  # Creates a Neo Block
segment = data.segments[0]  # Takes the first segment          

data2 = cortical_neurons_exc.get_data()
segment2 = data2.segments[0]
g = segment2.analogsignalarrays[0]
#plt.plot(np.meang)
#plt.show()

# Plot the spikes 

def plot_spiketrains(segment):
    for spiketrain in segment.spiketrains:
        y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
        plt.plot(spiketrain, y, '*b')
    plt.ylabel('Neuron number')
    plt.xlabel('Spikes')
        

#plot_spiketrains(segment)
#plt.show()

