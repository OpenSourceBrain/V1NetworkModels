from pyNN.utility import get_script_args, Timer
import numpy as np
import matplotlib.pyplot as plt 
import pyNN.space as space
from connector_functions import gabor_probability, lgn_to_cortical_connection, create_lgn_to_cortical
from connector_functions import create_cortical_to_cortical_connection, create_thalamocortical_connection
from analysis_functions import calculate_tuning
from plot_functions import plot_spiketrains
import cPickle

#############################

#simulator_name = get_script_args(1)[0]
#exec("import pyNN.%s as simulator" % simulator_name)
import pyNN.nest as simulator
#import pyNN.neuron as simulator

#############################
# Load LGN spikes and positions
#############################
directory = './data/'
format = '.cpickle'

## Load space 
layer = '_layer' + str(0)

polarity = '_on'
mark = 'positions'
positions_filename = directory + mark + polarity + layer + format
f1 = open(positions_filename, "rb")
positions_on = cPickle.load(f1)
f1.close()

polarity = '_off'
mark = 'positions'
positions_filename = directory + mark + polarity + layer + format
f1 = open(positions_filename, "rb")
positions_off = cPickle.load(f1)
f1.close()

## Load the spikes 

spikes_on = []
spikes_off   = []
N_lgn_layers = 4 

for layer in xrange(N_lgn_layers):

    #  Layer 1  
    layer = '_layer' + str(layer)
    
    polarity = '_on'
    mark = 'spike_train'
    spikes_filename = directory + mark + polarity + layer + format
    f2 = open(spikes_filename, 'rb')
    spikes_on.append(cPickle.load(f2))
    f2.close()
    
    polarity = '_off'
    mark = 'spike_train'
    spikes_filename = directory + mark + polarity + layer + format
    f2 = open(spikes_filename, 'rb')
    spikes_off.append(cPickle.load(f2))
    f2.close()


# Spike functions 


def spike_times_on(layer, spikes_on):
    return [simulator.Sequence(x) for x in spikes_on[layer]]

def spike_times_off(layer, spike_off):
    return [simulator.Sequence(x) for x in spikes_off[layer]]

#############################
## Network and Simulation parameters
#############################
Ncells_lgn = 30
Ncell_exc = 10
Ncell_inh = 10
#Ncell_exc = 40
#Ncell_inh = 20

t = 1000.0  # Simulation time

# Set the random set for reproducibility
seed = 1055
np.random.seed(seed)

# Has to be called at the beginning of the simulation
simulator.setup(timestep=0.1, min_delay=0.1, max_delay=5.0)


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
lgn_spikes_on_models = []
lgn_spikes_off_models = []


for layer in xrange(N_lgn_layers):
    model = simulator.SpikeSourceArray(spike_times=spike_times_on(layer, spikes_on))
    lgn_spikes_on_models.append(model)
    model = simulator.SpikeSourceArray(spike_times=spike_times_off(layer, spikes_off))
    lgn_spikes_off_models.append(model)

# LGN Popluations

lgn_on_populations = []
lgn_off_populations = []

for layer in xrange(N_lgn_layers):
    population = simulator.Population(Ncells_lgn**2, lgn_spikes_on_models[layer], structure=lgn_structure_on, label='LGN_on_layer_' +str(layer))
    lgn_on_populations.append(population)
    population = simulator.Population(Ncells_lgn**2, lgn_spikes_off_models[layer], structure=lgn_structure_off, label='LGN_off_layer_'+str(layer))
    lgn_off_populations.append(population)
    

## Cortical layer  

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
Vin = -70  # mV
t_fall_exc = 1.75  # ms
t_fall_inh = 5.27  # ms

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
cortical_neurons_exc = simulator.Population(Ncell_exc**2, excitatory_cell, structure=excitatory_structure,
                                            label='Excitatory layer')
cortical_neurons_inh = simulator.Population(Ncell_inh**2, inhibitory_cell, structure=inhibitory_structure,
                                            label='Inhibitory layer')

#############################
# Add background noise
#############################

correlated = False
noise_rate = 5800  # Hz
g_noise = 0.00089   # Microsiemens (0.89 Nanosiemens)
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

#lgn_on_populations = [lgn_neurons_on_0, lgn_neurons_on_1, lgn_neurons_on_2, lgn_neurons_on_3]


#lgn_off_populations = [lgn_neurons_off_0, lgn_neurons_off_1, lgn_neurons_off_2, lgn_neurons_off_3]



# Sample random phases and orientations 

phases_space = np.linspace(-180, 180, 20)
orientation_space = np.linspace(-40, 40, 9)

phases_exc = np.random.rand(Ncell_exc**2) * 360 * 0  # Phases continium
#phases_exc = np.random.choice(phases_space, Ncell_exc*2) # Phases discrete

orientations_exc = np.random.rand(Ncell_exc**2) * 180  # Orientations continium
orientations_exc = np.random.choice(orientation_space, Ncell_exc**2)   # Orientations discrete
#orientations_exc = np.linspace(-60, 60, Ncell_exc**2)  # Orientations ordered 

phases_inh = np.random.rand(Ncell_inh**2) * 360 * 0 
#phases_inh = np.random.choice(phases_space, Ncell_inh*2) # Phases discrete

#orientations_inh = np.random.rand(Ncell_inh**2) * np.pi
orientations_inh = np.random.choice(orientation_space, Ncell_inh**2)  # Orientations discrete

w = 0.8  # Spatial frequency
gamma = 1  # Aspect ratio
sigma = 1  # Decay ratio
g_exc = 0.00098  # microsiemens 
n_pick = 3  # Number of times to sample

polarity_on = 1
polarity_off = -1



if True:
    # Create the connections 
    for on_population, off_population in zip(lgn_on_populations, lgn_off_populations):

        create_thalamocortical_connection(on_population, cortical_neurons_exc, polarity_on, n_pick, g_exc, 
                                         sigma, gamma, w, phases_exc, orientations_exc, simulator)
        
        create_thalamocortical_connection(off_population, cortical_neurons_exc, polarity_off, n_pick, g_exc, 
                                         sigma, gamma, w, phases_exc, orientations_exc, simulator)
        
        create_thalamocortical_connection(on_population, cortical_neurons_inh, polarity_on, n_pick, g_exc, 
                                         sigma, gamma, w, phases_inh, orientations_inh, simulator)
        
        create_thalamocortical_connection(off_population, cortical_neurons_inh, polarity_off, n_pick, g_exc, 
                                         sigma, gamma, w, phases_inh, orientations_inh, simulator)

#############################
# Intracortical connections
#############################

# Connector parameters

n_pick = 10

orientation_sigma = 10   # Grades
phase_sigma = 10  # Grades
g_inh = 0.0083  # Nanosiemens 

# If true add cortical excitatory feedback (e -> e) and ( e -> i ) 
cortical_excitatory_feedback = False 

## We create inhibitory feed-forward connections ( i -> e)   

if False:
# Create list of connections 
    cortical_inh_exc_connections = create_cortical_to_cortical_connection(cortical_neurons_inh, cortical_neurons_exc,
                                                                          orientations_inh, phases_inh, orientations_exc,
                                                                          phases_exc, orientation_sigma, phase_sigma, g_inh,
                                                                          n_pick, target_type_excitatory=False)
    # Make the list a connector
    cortical_inh_exc_connector = simulator.FromListConnector(cortical_inh_exc_connections, column_names=["weight", "delay"])
    
    # Projections
    cortical_inh_exc_projection = simulator.Projection(cortical_neurons_inh, cortical_neurons_exc,
                                                       cortical_inh_exc_connector, receptor_type='inhibitory')

## Now we create the excitatory connections 
if cortical_excitatory_feedback:

    # Create list of connectors
    cortical_exc_exc_connections = create_cortical_to_cortical_connection(cortical_neurons_exc, cortical_neurons_exc,
                                                                          orientations_exc, phases_exc, orientations_exc,
                                                                          phases_exc, orientation_sigma, phase_sigma, g_exc,
                                                                          n_pick, target_type_excitatory=True)
    
    cortical_exc_inh_connections = create_cortical_to_cortical_connection(cortical_neurons_exc, cortical_neurons_inh,
                                                                          orientations_exc, phases_exc, orientations_inh,
                                                                          phases_inh, orientation_sigma, phase_sigma, g_exc,
                                                                          n_pick, target_type_excitatory=True)
    # Make the list a connector
    
    cortical_exc_exc_connector = simulator.FromListConnector(cortical_exc_exc_connections, column_names=["weight", "delay"])
    
    cortical_exc_inh_connector = simulator.FromListConnector(cortical_exc_inh_connections, column_names=["weight", "delay"])
    
    
    ## Projections
    cortical_exc_exc_projection = simulator.Projection(cortical_neurons_exc, cortical_neurons_exc,
                                                       cortical_exc_exc_connector, receptor_type='excitatory')
    
    cortical_exc_inh_projection = simulator.Projection(cortical_neurons_exc, cortical_neurons_inh,
                                                       cortical_exc_inh_connector, receptor_type='excitatory')


#############################n
# Recordings
#############################

cortical_neurons_exc.record(['gsyn_exc', 'gsyn_inh', 'v', 'spikes'])
#cortical_neurons_inh.record(['gsyn_exc', 'gsyn_inh', 'v', 'spikes'])
#cortical_neurons_exc.record('spikes', 'v')

#############################
# Run model
#############################
simulator.run(t)  # Run the simulations for t ms
simulator.end()

#############################
# Extract the data
#############################

#data = cortical_neurons_exc.get_data()
data = cortical_neurons_exc.get_data()  # Creates a Neo Block
segment = data.segments[0]  # Takes the first segment


## Show conductances 

if True:
    gexc = segment.analogsignalarrays[0]
    ginh = segment.analogsignalarrays[1]
    v = segment.analogsignalarrays[2]
    
    neuron = (orientations_exc == 0)
    plt.subplot(1, 2, 1)
    plt.plot(gexc.times, gexc[:,neuron], label='exc')
    plt.plot(ginh.times, ginh[:, neuron], label='inh')
     
    plt.subplot(1, 2, 2)
    plt.plot(v.times, v[:,neuron], label='v')
     
    plt.legend()
    plt.show()

#############################
# Extract the orientation dependence
#############################

def calculate_tuning(population, population_orientations, orientation_space):
    """
    Calculates and plots the mean rate of the cells in population as a function 
    of its orientations. The orientation_space is the space from where the orientations where 
    sampled 
    """
    mean_rate = np.zeros(orientation_space.size)
    
    for index, orientation in enumerate(orientation_space):
        mean_rate[index] = population[population_orientations == orientation].mean_spike_count()
    
    return mean_rate

rate = calculate_tuning(cortical_neurons_exc, orientations_exc, orientation_space)

plt.plot(orientation_space, rate)
plt.show()
rate = calculate_tuning(cortical_neurons_exc, orientations_exc, orientation_space)

plot_spiketrains(segment)
plt.show()
save = False

folder ='./output_data/'
voltage = '_v'
excitation = '_g_exc'
inhibition = '_g_inh'
format = '.pickle' 

filename_voltage = folder + voltage + format 
filename_excitation = folder + excitation + format  
filename_voltage = folder + excitation  + format 


if save:
    from neo.io import PickleIO
    
    io = PickleIO(filename=filename_voltage)
    cortical_neurons_exc.write_data(io, variables=['v'])
    
    io = PickleIO(filename=filename_voltage)
    cortical_neurons_exc.write_data(io, variables=['gsyn_exc'])
    
    io = PickleIO(filename=filename_voltage)
    cortical_neurons_exc.write_data(io, variables=['gsyn_inh'])
    
    
    