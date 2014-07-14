from pyNN.utility import get_script_args, Timer
import numpy as np
import matplotlib.pyplot as plt 
import pyNN.space as space
from connector_functions import gabor_probability
import cPickle

#############################

#simulator_name = get_script_args(1)[0]
#exec("import pyNN.%s as simulator" % simulator_name)
import pyNN.nest as simulator

#############################
# Load LGN spikes and positions
#############################
directory = './data/'
format = '.cpickle'
polarity = '_on'

layer = '_layer' + str(1)

mark = 'positions'
positions_filename = directory + mark + polarity + layer + format
f1 = open(positions_filename, "rb")
positions_on = cPickle.load(f1)
f1.close()

mark = 'spike_train'
spikes_filename = directory + mark + polarity + layer + format
f2 = open(spikes_filename, 'rb')
spikes_on = cPickle.load(f2)
f2.close()

polarity = '_off'

mark = 'positions'
positions_filename = directory + mark + polarity + layer + format
f1 = open(positions_filename, "rb")
positions_off = cPickle.load(f1)
f1.close()

mark = 'spike_train'
spikes_filename = directory + mark + polarity + layer + format
f2 = open(spikes_filename, 'rb')
spikes_off = cPickle.load(f2)
f2.close()

#############################
## Network and Simulation parameters
#############################
Ncells_lgn = 30
Ncell_exc = 40
Ncell_inh = 20

t = 1000.0  # Simulation time

# Set the random set for reproducibility
seed = 1055
np.random.seed(seed)

# Has to be called at the beginning of the simulation
simulator.setup(timestep=0.1, min_delay=0.1, max_delay=5.0)

def spike_times_on(i):
    '''
    Test function
    '''

    A = []
    for k in range(len(i)):
        A.append(spikes_on[k])

    return A

def spike_times_on(i):
    return spikes_on

def spike_times_off(i):
    return spikes_off


# Spatial structure of on LGN cells
# On cells
x0, y0 = positions_on[0] # Take out the lower corner from the positions
x_end, y_end = positions_on[-1]
x1, dummy = positions_on[Ncells_lgn]
dummy, y1 = positions_on[1]
dx = x1 - x0
dy = y1 - y0
lx = x_end - x0
ly = y_end - y0


lgn_structure_on = space.Grid2D(aspect_ratio=1, x0=x0, y0=y0, dx=dx, dy=dy, z=0)

# Off cells
x0, y0 = positions_off[0] # Take out the lower corner from the positions


x_end, y_end = positions_off[-1]
x1, dummy = positions_off[Ncells_lgn]
dummy, y1 = positions_off[1]
dx = x1 - x0
dy = y1 - y0
lx = x_end - x0
ly = y_end - y0

lgn_structure_off = space.Grid2D(aspect_ratio=1, x0=x0, y0=y0, dx=dx, dy=dy, z=0)

# Spikes for LGN populations
lgn_spikes_on_model = simulator.SpikeSourceArray(spike_times=spike_times_on)
lgn_spikes_off_model = simulator.SpikeSourceArray(spike_times=spike_times_off)

# LGN Popluations
lgn_neurons_on = simulator.Population(Ncells_lgn **2, lgn_spikes_on_model, structure=lgn_structure_on, label='LGN_on')
lgn_neurons_off = simulator.Population(Ncells_lgn **2, lgn_spikes_off_model, structure=lgn_structure_off, label='LGN_off')

# Spatial structure of cortical cells
lx = 0.75
ly = 0.75
x0 = -lx / 2
y0 = -ly / 2
dx = lx / Ncell_exc
dy = ly / Ncell_exc

excitatory_structure = space.Grid2D(aspect_ratio=1, x0=x0, y0=y0, dx=dx, dy=dy, z=0)
inhibitory_structure = space.Grid2D(aspect_ratio=1, x0=0, y0=0, dx=2*dx, dy=2*dy, z=0)

# Cortical parameters
Vth = -52.5
delay = 2.00


# Cortical Population
cortical_neurons_exc = simulator.Population(1, simulator.IF_cond_exp())
#cortical_neurons_exc = simulator.Population(Ncell_exc**2, simulator.IF_cond_exp(), structure=excitatory_structure)
cortical_neurons_inh = simulator.Population(Ncell_inh**2, simulator.IF_cond_exp(), structure=inhibitory_structure)

#############################
# Add background noise
#############################

correlated = True
noise_rate = 5800 # Hz
g_noise = 0.00089  # Microsiemens
noise_delay = 1 # ms
noise_model = simulator.SpikeSourcePoisson(rate=noise_rate)
noise_syn = simulator.StaticSynapse(weight=g_noise, delay=noise_delay)


if correlated:
    # If correlated is True all the cortical neurons receive noise from the same cell.
    noise_population = simulator.Population(1, noise_model, label='Background Noise')
    background_noise_to_exc = simulator.Projection(noise_population, cortical_neurons_exc,
                                                   simulator.AllToAllConnector(), noise_syn)
    background_noise_to_inh = simulator.Projection(noise_population, cortical_neurons_inh,
                                                   simulator.AllToAllConnector(), noise_syn)
else:
    # If correlated is False, all cortical neurons receive independent noise
    noise_population_exc = simulator.Population(Ncell_exc**2, noise_model, label='Background Noise to Exc')
    noise_population_inh = simulator.Population(Ncell_inh**2, noise_model, label='Background Noise to Inh')
    background_noise_to_exc = simulator.Projection(noise_population_exc, cortical_neurons_exc,
                                                   simulator.OneToOneConnector(), noise_syn)
    background_noise_to_inh = simulator.Projection(noise_population_inh, cortical_neurons_inh,
                                                   simulator.OneToOneConnector(), noise_syn)

#############################
## Thalamo-Cortical connections
#############################

def lgn_to_cortical_connection(connect_to_neuron, connections, lgn_neurons, n_pick, g_exc, polarity):
    """
    Connections from the thalamus to the cell connecto_to_neuron
    """

    for lgn_neuron in lgn_neurons:
            # Extract position
            x, y = lgn_neuron.position[0:2]
            #print 'x, y', x, y
            # Calculate the gabbor probability
            probability = polarity * gabor_probability(x, y, sigma, gamma, phi, w, theta)
            probability = np.sum(np.random.rand(n_pick) < probability) # Samples
            #print probability
            synaptic_weight = (g_exc / n_pick) * probability
            #print synaptic_weight
            neuron_index = lgn_neurons.id_to_index(lgn_neuron)

            # The format of the connector list should be pre_neuron, post_neuron, w, tau_delay
            if synaptic_weight > 0 :
                connections.append((neuron_index, connect_to_neuron, synaptic_weight, 0.1))

## Create the connector
exc_connections = []
inh_connections = []

for cortical_neuron in cortical_neurons_exc:

    w = 0.8  # Spatial frequency
    random_phase = np.random.rand() * 2
    phi = random_phase * np.pi  # Phase
    gamma = 1  # Aspect ratio
    sigma = 1  # Decay ratio
    random_orientation = np.random.rand()
    theta = random_orientation * np.pi # Orientation

    g_exc = 0.98 
    n_pick = 3  # Number of times to sample

    connect_to_neuron = cortical_neurons_exc.id_to_index(cortical_neuron)  # Connects to this neuron
    print 'connect to neuron', connect_to_neuron
    lgn_to_cortical_connection(connect_to_neuron, exc_connections, lgn_neurons_on, n_pick, g_exc, 1)
    lgn_to_cortical_connection(connect_to_neuron, inh_connections, lgn_neurons_off, n_pick, g_exc, -1)

# Make the list a connector
exc_connector = simulator.FromListConnector(exc_connections, column_names=["weight", "delay"])
inh_connector = simulator.FromListConnector(inh_connections, column_names=["weight", "delay"])

# Synapses
syn1 = simulator.StaticSynapse(weight=1, delay=0.5)
syn2 = simulator.StaticSynapse(weight=1, delay=3)

excitatory_connections = simulator.Projection(lgn_neurons_on, cortical_neurons_exc, exc_connector,
                                              syn1, receptor_type='excitatory')

inhibitory_connections = simulator.Projection(lgn_neurons_on, cortical_neurons_exc, inh_connector,
                                              syn1, receptor_type='inhibitory')


#############################
# Intracortical connections
#############################


#############################n
# Recordings
#############################

lgn_neurons_on.record('spikes')
cortical_neurons_exc.record(['gsyn_exc', 'gsyn_inh'])


#############################
# Run model
#############################
simulator.run(t)  # Run the simulations for t ms
simulator.end()

#############################
# Extract the data
#############################

data = lgn_neurons_on.get_data()  # Creates a Neo Block
segment = data.segments[0]  # Takes the first segment          

data2 = cortical_neurons_exc.get_data()
segment2 = data2.segments[0]
g_exc = segment2.analogsignalarrays[0]
g_inh = segment2.analogsignalarrays[1]
#plt.plot(g_exc.times, np.mean(g_exc, axis=1))
#plt.plot(g_inh.times, -np.mean(g_inh, axis=1))

plt.plot(g_exc.times, g_exc[:, 0])
plt.plot(g_inh.times, g_inh[:, 0])

plt.show()

# Plot the spikes 

def plot_spiketrains(segment):
    for spiketrain in segment.spiketrains:
        y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
        plt.plot(spiketrain, y, '*b')
    plt.ylabel('Neuron number')
    plt.xlabel('Spikes')
        

#plot_spiketrains(segment)
#plt.show()

