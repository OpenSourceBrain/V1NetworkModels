from pyNN.utility import get_script_args, Timer
import numpy as np
import matplotlib.pyplot as plt 
import pyNN.space as space
from connector_functions import gabor_probability, lgn_to_cortical_connection, create_lgn_to_cortical
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
Ncell_exc = 10
Ncell_inh = 5

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
lgn_neurons_on = simulator.Population(Ncells_lgn**2, lgn_spikes_on_model, structure=lgn_structure_on, label='LGN_on')
lgn_neurons_off = simulator.Population(Ncells_lgn**2, lgn_spikes_off_model, structure=lgn_structure_off, label='LGN_off')

# Spatial structure of cortical cells
lx = 0.75
ly = 0.75
x0 = -lx / 2
y0 = -ly / 2
dx = lx / (Ncell_exc - 1)
dy = ly / (Ncell_exc - 1)

excitatory_structure = space.Grid2D(aspect_ratio=1, x0=x0, y0=y0, dx=dx, dy=dy, z=0)
inhibitory_structure = space.Grid2D(aspect_ratio=1, x0=0, y0=0, dx=2*dx, dy=2*dy, z=0)

# Cortical parameters

# Common
Vth = -52.5  # mV
delay = 2.00  # ms

# Conductances
Vex = 0  # mV
Vin = -70 # mV
t_fall_exc = 1.75 # mV
t_fall_inh = 5.27 # mV


# Excitatory
C_exc = 0.500  # Nanofarads (500 Picofarads)
g_leak_exc = 0.025  # Microsiemens (25 Nanosiemens)
t_refrac_exc = 1.5  # ms
v_leak_exc = -73.6  # mV
v_reset_exc = -56.6  # mV
t_m_exc = C_exc / g_leak_exc  # Membrane time constant

# Inhibitory
C_inh = 0.214  # Nanofarads (214 Picofarads)
g_leak_inh = 0.018  # Microsiemens (18 Nanosiemens)
v_leak_inh = -81.6  # mV
v_reset_inh = -57.8  # mV
t_refrac_inh = 1.0  # ms
t_m_inh = C_inh / g_leak_inh  # Membrane time constant

# Cortical cell models
excitatory_cell = simulator.IF_cond_exp(tau_refrac=t_refrac_exc, cm=C_exc, tau_syn_E=t_fall_exc, v_rest=v_leak_exc,
                                        tau_syn_I=t_fall_inh, tau_m=t_m_exc, e_rev_E=Vex, e_rev_I=Vin, v_thresh=Vth,
                                        v_reset=v_reset_exc)

inhibitory_cell = simulator.IF_cond_exp(tau_refrac=t_refrac_inh, cm=C_inh, tau_syn_E=t_fall_exc, v_rest=v_leak_inh,
                                        tau_syn_I=t_fall_inh, tau_m=t_m_inh, e_rev_E=Vex, e_rev_I=Vin, v_thresh=Vth,
                                        v_reset=v_reset_inh)


# Cortical Population
cortical_neurons_exc = simulator.Population(1, excitatory_cell)
cortical_neurons_exc = simulator.Population(Ncell_exc**2, excitatory_cell, structure=excitatory_structure)
cortical_neurons_inh = simulator.Population(Ncell_inh**2, inhibitory_cell, structure=inhibitory_structure)

#############################
# Add background noise
#############################

correlated = True
noise_rate = 5800  # Hz
g_noise = 0.00089  # Microsiemens (0.89 Nanosiemens)
noise_delay = 1  # ms
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


## Create the connector
exc_connections = []
inh_connections = []

phases_exc = np.random.rand(Ncell_exc**2) * 2 * np.pi
orientations_exc = np.random.rand(Ncell_exc**2) * np.pi

phases_inh = np.random.rand(Ncell_inh**2) * 2 * np.pi
orientations_inh = np.random.rand(Ncell_inh**2) * np.pi
w = 0.8  # Spatial frequency
gamma = 1  # Aspect ratio
sigma = 1  # Decay ratio
g_exc = 0.00098  # microsiemens
n_pick = 3  # Number of times to sample



polarity_on = 1
polarity_off = -1
exc_connections = create_lgn_to_cortical(lgn_neurons_on, cortical_neurons_exc, polarity_on,  n_pick, g_exc,
                                         sigma, gamma, phases_exc, w, orientations_exc)

inh_connections = create_lgn_to_cortical(lgn_neurons_off, cortical_neurons_exc, polarity_off,  n_pick, g_exc,
                                         sigma, gamma, phases_exc, w, orientations_exc)



# Make the list a connector
exc_connector = simulator.FromListConnector(exc_connections, column_names=["weight", "delay"])
inh_connector = simulator.FromListConnector(inh_connections, column_names=["weight", "delay"])

# Synapses
syn1 = simulator.StaticSynapse(weight=1, delay=0.5)
syn2 = simulator.StaticSynapse(weight=1, delay=3)

excitatory_connections = simulator.Projection(lgn_neurons_on, cortical_neurons_exc, exc_connector,
                                              receptor_type='excitatory')

inhibitory_connections = simulator.Projection(lgn_neurons_on, cortical_neurons_exc, inh_connector,
                                              receptor_type='inhibitory')


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

