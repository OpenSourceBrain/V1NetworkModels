from pyNN.utility import get_script_args, Timer
import numpy as np
import matplotlib.pyplot as plt 
import pyNN.space as space
from connector_functions import gabor_probability, lgn_to_cortical_connection, create_lgn_to_cortical
from connector_functions import create_cortical_to_cortical_connection, create_thalamocortical_connection
from analysis_functions import calculate_tuning, visualize_conductances, visualize_conductances_and_voltage
from analysis_functions import conductance_analysis
from plot_functions import plot_spiketrains
import cPickle

#############################

#simulator_name = get_script_args(1)[0]
#exec("import pyNN.%s as simulator" % simulator_name)
import pyNN.nest as simulator
#import pyNN.neuron as simulator

#############################
## Network and Simulation parameters
#############################

contrast = 0.50

# Layer size and dimensions 

Ncells_lgn = 30
Ncell_exc = 40
Ncell_inh = 20

factor = 1.0
Ncell_exc = int(factor * Ncell_exc)
Ncell_inh = int(factor * Ncell_inh)


N_lgn_layers = 1
# If True add cortical excitatory feedback (e -> e) and ( e -> i ) 
cortical_excitatory_feedback = False
# If True add feedforward inhibition ( i -> e )  
feed_forward_inhibition = True
# If True add cortical noise 
background_noise = True
correlated_noise = False
# If True create connections from the thalamus to the cortex
thalamo_cortical_connections = True 

# Simulation time
t = 1000.0 
# Set the random set for reproducibility
seed = 1055
np.random.seed(seed)

# Has to be called at the beginning of the simulation
#simulator.setup(timestep=0.1, min_delay=0.1, max_delay=5.0)
simulator.setup(timestep=1, min_delay=1, max_delay=5.0)


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
spikes_off = []

for layer in xrange(N_lgn_layers):

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


# Spike functions 


def spike_times_on(layer, spikes_on):
    return [simulator.Sequence(x) for x in spikes_on[layer]]

def spike_times_off(layer, spike_off):
    return [simulator.Sequence(x) for x in spikes_off[layer]]


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

noise_rate = 5800  # Hz
g_noise = 0.00089   # Microsiemens (0.89 Nanosiemens)
noise_delay = 1  # ms
noise_model = simulator.SpikeSourcePoisson(rate=noise_rate)
noise_syn = simulator.StaticSynapse(weight=g_noise, delay=noise_delay)

if background_noise:

    if correlated_noise:
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

# Sample random phases and orientations 

phases_space = np.linspace(-180, 180, 19)
#orientation_space = np.linspace(0, 90, 10)
#orientation_space = np.linspace(-90, 90, 19)
orientation_space = np.linspace(-60, 60, 13)

phases_exc = np.random.rand(Ncell_exc**2) * 360 * 0  # Phases continium
#phases_exc = np.random.choice(phases_space, Ncell_exc*2) # Phases discrete

orientations_exc = np.random.rand(Ncell_exc**2) * 180  # Orientations continium
orientations_exc = np.random.choice(orientation_space, Ncell_exc**2) * 1  # Orientations discrete
#orientations_exc = np.linspace(-60, 60, Ncell_exc**2)  # Orientations ordered 

phases_inh = np.random.rand(Ncell_inh**2) * 360 * 0
phases_inh = 180 - np.random.choice(phases_exc, Ncell_inh**2)
#phases_inh = np.random.choice(phases_space, Ncell_inh*2) # Phases discrete

#orientations_inh = np.random.rand(Ncell_inh**2) * np.pi
orientations_inh = np.random.choice(orientation_space, Ncell_inh**2) * 1  # Orientations discrete
#orientations_inh = np.random.choice(orientations_exc, Ncell_inh**2) # Sample from the orientaitons of excitatory cells

w = 0.8  # Spatial frequency
gamma = 1  # Aspect ratio
sigma = 1  # Decay ratio
g_exc = (4.0 / N_lgn_layers) * 0.00098  # microsiemens
n_pick = 3  # Number of times to sample

polarity_on = 1
polarity_off = -1



if thalamo_cortical_connections:
    # Create thalamo-cortical connections 
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

orientation_sigma = 20   # Grades
phase_sigma = 20  # Grades
g_inh = 0.00083 * (3 / factor)  # Nanosiemens



## We create inhibitory feed-forward connections ( i -> e)   

if feed_forward_inhibition:
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

cortical_neurons_exc.record(['gsyn_exc', 'gsyn_inh','v', 'spikes'])
#cortical_neurons_exc.record(['gsyn_exc', 'gsyn_inh','v', 'spikes'])
cortical_neurons_inh.record(['gsyn_exc', 'gsyn_inh', 'v', 'spikes'])
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

data = cortical_neurons_inh.get_data()
segment2 = data.segments[0]

## Show conductances 


test_orientation = orientations_exc[0]
test_orientation = 0
neurons = (orientations_exc == test_orientation)
print 'test orientation', test_orientation
#visualize_conductances_and_voltage(segment, neurons)


if False:
    DC, F1 = conductance_analysis(cortical_neurons_exc, segment, orientation_space)
    plt.plot(orientation_space, DC, label='DC')
    plt.plot(orientation_space, DC + F1, label='DC + F1')
    plt.legend()
    plt.show()

#############################
# Extract the orientation dependence
#############################


if True:
    visualize_conductances(segment, neurons)
    plt.show()


    rate = calculate_tuning(cortical_neurons_exc, orientations_exc, orientation_space)

    plt.plot(orientation_space, rate)
    plt.xlabel('Orientation')
    plt.ylabel('Firing Rate')
    plt.show()


if True:  # Spikes analysis

    plt.subplot(2, 1, 1)

    plot_spiketrains(segment)

    plt.subplot(2, 1, 2)
    plot_spiketrains(segment2)

    plt.show()


save = False
folder ='./output_data/'
voltage = '_v'
excitation = '_g_exc'
inhibition = '_g_inh'
format = '.pickle' 

filename_voltage = folder + voltage + format 
filename_excitation = folder + excitation + format  
filename_voltage = folder + excitation + format


if save:
    from neo.io import PickleIO
    
    io = PickleIO(filename=filename_voltage)
    cortical_neurons_exc.write_data(io, variables=['v'])
    
    io = PickleIO(filename=filename_voltage)
    cortical_neurons_exc.write_data(io, variables=['gsyn_exc'])
    
    io = PickleIO(filename=filename_voltage)
    cortical_neurons_exc.write_data(io, variables=['gsyn_inh'])
    
    
    