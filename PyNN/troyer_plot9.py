from pyNN.utility import get_script_args, Timer
import numpy as np
import matplotlib.pyplot as plt
from connector_functions import load_positions, load_lgn_spikes, return_lgn_starting_coordinates
import pyNN.space as space
from connector_functions import create_cortical_to_cortical_connection
from connector_functions import normalize_connection_list
from connector_functions import create_cortical_to_cortical_connection_corr
from connector_functions import create_thalamocortical_connection
from analysis_functions import calculate_tuning, visualize_conductances, visualize_conductances_and_voltage
from analysis_functions import conductance_analysis
from plot_functions import plot_spiketrains

#############################

simulator = get_script_args(1)[0]
exec("import pyNN.%s as simulator" % simulator)
#import pyNN.nest as simulator
#import pyNN.neuron as simulator

timer = Timer()

#############################
##  Parameters
#############################

# ============== Network and simulation parameters =================

contrast = 0.50  # Contrast used (possible range available in ./data)

Nside_lgn = 30  # N_lgn x N_lgn is the size of the LGN
Nside_exc = 40  # N_exc x N_exc is the  size of the cortical excitatory layer
Nside_inh = 20  # N_inh x N_inh is the size of the cortical inhibitory layer

factor = 1  # Reduction factor

assert 0 < factor <= 1, 'factor out of (0,1] range'

Nside_exc = int(factor * Nside_exc)
Nside_inh = int(factor * Nside_inh)

Ncell_lgn = Nside_lgn * Nside_lgn
Ncell_exc = Nside_exc ** 2
Ncell_inh = Nside_inh ** 2

N_lgn_layers = 1

## Main connections
thalamo_cortical_connections = True  # If True create connections from the thalamus to the cortex
feed_forward_inhibition = True  # If True add feed-forward inhibition ( i -> e )
cortical_excitatory_feedback = False  # If True add cortical excitatory feedback (e -> e) and ( e -> i )
background_noise = True  # If True add cortical noise
correlated_noise = False  # Makes the noise correlated

# Save
save_voltage_and_conductances = False
save_orientation_response = False

# Plot
plot_conductance_analysis = True
plot_spike_analysis = True
plot_orientation_analysis = True

# Simulation time
t = 1000.0
# Set the random set for reproducibility
seed = 1055
np.random.seed(seed)


# ============== Parameters of the thalamo-cortical connection =============

## Orientation and phase space (We sample from here for each cell)

phases_space = np.linspace(-180, 180, 19)
#orientation_space = np.linspace(0, 90, 10)
orientation_space = np.linspace(-90, 90, 19)
#orientation_space = np.linspace(0, 180, 19)
#orientation_space = np.linspace(-60, 60, 13)

## Assign excitatory phases and orientations

phases_exc = np.random.rand(Ncell_exc) * 360 * 0  # Phases continium
#phases_exc = np.random.choice(phases_space, Ncell_exc)  # Phases discrete
#orientations_exc = np.random.rand(Ncell_exc) * 180  # Orientations continium
orientations_exc = np.random.choice(orientation_space, Ncell_exc)  # Orientations discrete
#orientations_exc = np.linspace(-60, 60, Ncell_exc)  # Orientations ordered

## Assign inhibitory phases and orientations

#phases_inh = np.random.rand(Ncell_inh) * 360 * 0
phases_inh = 180 - np.random.choice(phases_exc, Ncell_inh)
#phases_inh = np.random.choice(phases_space, Ncell_inh)  # Phases discrete
#orientations_inh = np.random.rand(Ncell_inh) * np.pi
orientations_inh = np.random.choice(orientation_space, Ncell_inh)   # Orientations discrete
#orientations_inh = np.random.choice(orientations_exc, Ncell_inh) # Sample from the orientaitons of excitatory cells

## Gabor function parameters

w = 0.8  # Spatial frequency
gamma = 2  # Aspect ratio
sigma = 1  # Decay ratio
g_exc = (4.0 / N_lgn_layers) * 0.00098  # microsiemens
g_exc = (4.0 / N_lgn_layers) * 0.0021   # microsiemens
n_pick = 3  # Number of times to sample
lgn_delay = 0.1

polarity_on = 1
polarity_off = -1

# ============== Background noise parameters ============================

noise_rate = 5800  # Hz
g_noise = 0.00089   # Microsiemens (0.89 Nanosiemens)
noise_delay = 1  # ms

# =============  Cortical layers' parameters =========================

# Common to both layers
Vth = -52.5  # mV
Vth = -55.0
delay = 2.00  # ms
Vex = 100  # mV
Vin = -70  # mV
t_fall_exc = 1.75  # ms
t_fall_inh = 5.27  # ms

# Excitatory layer
C_exc = 0.500  # Nanofarads (500 Picofarads)
g_leak_exc = 0.025  # Microsiemens (25 Nanosiemens)
t_refrac_exc = 1.5  # ms
v_leak_exc = -73.6  # mV
v_reset_exc = -56.6  # mV
t_m_exc = C_exc / g_leak_exc  # Membrane time constant

# Inhibitory inhibitory layer
C_inh = 0.214  # Nanofarads (214 Picofarads)
g_leak_inh = 0.018  # Microsiemens (18 Nanosiemens)
v_leak_inh = -81.6  # mV
v_reset_inh = -57.8  # mV
t_refrac_inh = 1.0  # ms
t_m_inh = C_inh / g_leak_inh  # Membrane time constant

# ============= Cortical connections' parameters =======

n_pick = 10
orientation_sigma = 10   # Grades
phase_sigma = 10  # Grades
g_inh = 0.00083 * (2 / factor)
#g_inh = 0.0083 * 35 # * (2 / factor)  # Nanosiemens
cortical_delay = 0.1


# ================= Simulation time ==================
dt = 1.0  # Simulation's time step
delay_min = 1.0  # Minimum delay
delay_max = 5.0  # Maximum delay

#############################
# Build the Network
#############################

# Has to be called at the beginning of the simulation
simulator.setup(timestep=dt, min_delay=delay_min, max_delay=delay_max)

timer.start()  # start timer on construction

# ================== LGN ========================

# Load LGN positions
positions_on, positions_off = load_positions()

## Load the spikes
spikes_on, spikes_off = load_lgn_spikes(contrast, N_lgn_layers)

# Spike functions
def spike_times(simulator, layer, spikes_file):
    return [simulator.Sequence(x) for x in spikes_file[layer]]

# Spatial structure of on LGN cells
# On cells

x0, y0, dx, dy = return_lgn_starting_coordinates(positions_on, Nside_lgn)
lgn_structure_on = space.Grid2D(aspect_ratio=1, x0=x0, y0=y0, dx=dx, dy=dy, z=0)

# Off cells

x0, y0, dx, dy = return_lgn_starting_coordinates(positions_off, Nside_lgn)
lgn_structure_off = space.Grid2D(aspect_ratio=1, x0=x0, y0=y0, dx=dx, dy=dy, z=0)

# Cells models for the LGN spikes (SpikeSourceArray)
lgn_spikes_on_models = []
lgn_spikes_off_models = []


for layer in xrange(N_lgn_layers):
    model = simulator.SpikeSourceArray(spike_times=spike_times(simulator, layer, spikes_on))
    lgn_spikes_on_models.append(model)
    model = simulator.SpikeSourceArray(spike_times=spike_times(simulator, layer, spikes_off))
    lgn_spikes_off_models.append(model)

# LGN Populations

lgn_on_populations = []
lgn_off_populations = []

for layer in xrange(N_lgn_layers):
    population = simulator.Population(Ncell_lgn, lgn_spikes_on_models[layer], structure=lgn_structure_on,
                                      label='LGN_on_layer_' + str(layer))
    lgn_on_populations.append(population)
    population = simulator.Population(Ncell_lgn, lgn_spikes_off_models[layer], structure=lgn_structure_off,
                                      label='LGN_off_layer_' + str(layer))
    lgn_off_populations.append(population)


# =================== Cortical Layers =======================

# Spatial structure of cortical cells
lx = 0.75
ly = 0.75
x0 = -lx / 2
y0 = -ly / 2
dx = lx / (Ncell_exc - 1)
dy = ly / (Ncell_exc - 1)

excitatory_structure = space.Grid2D(aspect_ratio=1, x0=x0, y0=y0, dx=dx, dy=dy, z=0)
inhibitory_structure = space.Grid2D(aspect_ratio=1, x0=0, y0=0, dx=2*dx, dy=2*dy, z=0)


# Cortical cell models
excitatory_cell = simulator.IF_cond_exp(tau_refrac=t_refrac_exc, cm=C_exc, tau_syn_E=t_fall_exc, v_rest=v_leak_exc,
                                        tau_syn_I=t_fall_inh, tau_m=t_m_exc, e_rev_E=Vex, e_rev_I=Vin, v_thresh=Vth,
                                        v_reset=v_reset_exc)

inhibitory_cell = simulator.IF_cond_exp(tau_refrac=t_refrac_inh, cm=C_inh, tau_syn_E=t_fall_exc, v_rest=v_leak_inh,
                                        tau_syn_I=t_fall_inh, tau_m=t_m_inh, e_rev_E=Vex, e_rev_I=Vin, v_thresh=Vth,
                                        v_reset=v_reset_inh)

# Cortical Population
cortical_neurons_exc = simulator.Population(Ncell_exc, excitatory_cell, structure=excitatory_structure,
                                            label='Excitatory layer')

cortical_neurons_exc.initialize(v=v_reset_exc)

cortical_neurons_inh = simulator.Population(Ncell_inh, inhibitory_cell, structure=inhibitory_structure,
                                            label='Inhibitory layer')

cortical_neurons_inh.initialize(v=v_reset_inh)

# ============== Cortical background noise =====================

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
        noise_population_exc = simulator.Population(Ncell_exc, noise_model, label='Background Noise to Exc')
        noise_population_inh = simulator.Population(Ncell_inh, noise_model, label='Background Noise to Inh')
        background_noise_to_exc = simulator.Projection(noise_population_exc, cortical_neurons_exc,
                                                       simulator.OneToOneConnector(), noise_syn)
        background_noise_to_inh = simulator.Projection(noise_population_inh, cortical_neurons_inh,
                                                       simulator.OneToOneConnector(), noise_syn)


# ========== Thalamo-Cortical connections ==============

if thalamo_cortical_connections:
    # Create thalamo-cortical connections
    for on_population, off_population in zip(lgn_on_populations, lgn_off_populations):

        create_thalamocortical_connection(on_population, cortical_neurons_exc, polarity_on, n_pick, g_exc, lgn_delay,
                                         sigma, gamma, w, phases_exc, orientations_exc, simulator)

        create_thalamocortical_connection(off_population, cortical_neurons_exc, polarity_off, n_pick, g_exc, lgn_delay,
                                         sigma, gamma, w, phases_exc, orientations_exc, simulator)

        create_thalamocortical_connection(on_population, cortical_neurons_inh, polarity_on, n_pick, g_exc, lgn_delay,
                                         sigma, gamma, w, phases_inh, orientations_inh, simulator)

        create_thalamocortical_connection(off_population, cortical_neurons_inh, polarity_off, n_pick, g_exc, lgn_delay,
                                         sigma, gamma, w, phases_inh, orientations_inh, simulator)


# ============ Intracortical connections ===================

## We create inhibitory feed-forward connections ( i -> e)

if feed_forward_inhibition:
    # Create list of connections
    cortical_inh_exc_connections = create_cortical_to_cortical_connection(cortical_neurons_inh, cortical_neurons_exc,
                                                                          orientations_inh, phases_inh, orientations_exc,
                                                                          phases_exc, orientation_sigma, phase_sigma, g_inh,
                                                                          cortical_delay, n_pick, target_type_excitatory=False)
    # Normalize the list
    cortical_inh_exc_connections = normalize_connection_list(cortical_inh_exc_connections)

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
                                                                          cortical_delay, n_pick, target_type_excitatory=True)

    # Normalize the list
    cortical_exc_exc_connections = normalize_connection_list(cortical_exc_exc_connections)

    cortical_exc_inh_connections = create_cortical_to_cortical_connection(cortical_neurons_exc, cortical_neurons_inh,
                                                                          orientations_exc, phases_exc, orientations_inh,
                                                                          phases_inh, orientation_sigma, phase_sigma, g_exc,
                                                                          cortical_delay, n_pick, target_type_excitatory=True)
    # Normalize the list
    cortical_exc_inh_connections = normalize_connection_list(cortical_exc_inh_connections)


    # Make the list a connector

    cortical_exc_exc_connector = simulator.FromListConnector(cortical_exc_exc_connections, column_names=["weight", "delay"])

    cortical_exc_inh_connector = simulator.FromListConnector(cortical_exc_inh_connections, column_names=["weight", "delay"])

    ## Projections
    if len(cortical_exc_exc_connections) != 0:
        cortical_exc_exc_projection = simulator.Projection(cortical_neurons_exc, cortical_neurons_exc,
                                                           cortical_exc_exc_connector, receptor_type='excitatory')

    if len(cortical_exc_inh_connections) != 0:

        cortical_exc_inh_projection = simulator.Projection(cortical_neurons_exc, cortical_neurons_inh,
                                                           cortical_exc_inh_connector, receptor_type='excitatory')



#############################n
# Recordings
#############################

cortical_neurons_exc.record(['v', 'spikes'])

# read out time used for building
build_time = timer.elapsedTime()

#############################
# Run model and print information
#############################
simulator.run(t)  # Run the simulations for t ms
simulation_time = timer.elapsedTime()

print 'Construction time', build_time
print 'Simulation time', simulation_time

simulator.end()

#############################
# Extract the data
#############################

#data = cortical_neurons_exc.get_data()
data = cortical_neurons_exc.get_data()  # Creates a Neo Block
segment = data.segments[0]  # Takes the first segment
v = segment.analogsignalarrays[0]

#############################
# Do the analysis
#############################

# ============== Voltage traces =================

orientations_to_analyze = np.linspace(0, 80, 9)
neurons = np.zeros(orientations_to_analyze.size)

for index, orientation in enumerate(orientations_to_analyze):
    aux = np.where(orientations_exc == orientation)[0][0]
    neurons[index] = aux

neurons = neurons.astype('int')

for index, neuron in enumerate(neurons):

    plt.subplot(3, 3, index + 1)
    plt.plot(v.times, v[:, neuron], label='theta=' + str(orientations_to_analyze[index]))
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.ylim([-70, -50])
    plt.legend()

print 'Vth', Vth
plt.show()
