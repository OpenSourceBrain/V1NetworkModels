import pyNN.nest as simulator
import pyNN.space as space
import matplotlib.pyplot as plt
import numpy as np


n_retina = 10
N_retina = n_retina * n_retina

n_lgn = 1
N_lgn = n_lgn * n_lgn
  
t = 200.0  # Simulation time

# Has to be called at the beginning of the simulation
simulator.setup(timestep=0.1, min_delay=0.1, max_delay=10)  # What are delays and max delay?

# Neuron Model's parameters

i_offset = 0
i_offset = 0
R = 20
tau_m = 20.0
tau_refractory = 1
v_thresh = 0
v_rest = -60
tau_syn_E = 5.0
tau_syn_I = 5.0
cm = tau_m / R

# It seems that the resting potential is -65 for every neuron
model = simulator.IF_curr_exp(cm=cm, i_offset=i_offset, tau_m=tau_m, tau_refrac=tau_refractory, tau_syn_E=tau_syn_E,
                              tau_syn_I=tau_syn_I, v_reset=v_rest, v_thresh=v_thresh)

# Spatial structure
retinal_structure = space.Grid2D(aspect_ratio=1, dx=1.0, dy=1.0, z=0)

# Populations
retinal_neurons = simulator.Population(N_retina, model, structure=retinal_structure, label='Retina')
lgn_neurons = simulator.Population(N_lgn, model, structure=retinal_structure, label='LGN')

# Initialize populations
v_random = np.random.rand(N_retina) * (v_thresh - v_rest) + v_rest
#retinal_neurons.initialize(v=v_random) # Random Initialization

v_random = np.random.rand(N_lgn) * (v_thresh - v_rest) + v_rest
lgn_neurons.initialize(v=v_random) 

# Synapses, Connections and Projections
# Synapses
exc_synapse = simulator.StaticSynapse(weight=20.0, delay=0.1)  # The synapses
inh_synapse = simulator.StaticSynapse(weight=10.0, delay=0.1)  # The synapses

# Connectors
# Position dependent probability connector 
DDPC = simulator.DistanceDependentProbabilityConnector
exc_connector = DDPC('d < 3')
inh_connector = DDPC('d < 7 and 3 < d')
# Projections

space = space.Space(axes=None, periodic_boundaries=((0, N_retina), (0, N_retina), None))

exc_projection = simulator.Projection(retinal_neurons, lgn_neurons, connector=exc_connector, synapse_type=exc_synapse,
                                      receptor_type='excitatory', space=space, label='excitatory connections')

inh_projection = simulator.Projection(retinal_neurons, lgn_neurons, connector=inh_connector, synapse_type=inh_synapse,
                                      receptor_type='inhibitory', space=space, label='inhibitory connections')


# Current
noise = simulator.NoisyCurrentSource(mean=0, stdev=8.0, start=1.0, stop=400.0, dt=1.0)
for i in xrange(N_retina):
    noise.inject_into(retinal_neurons[[i]])

#current = simulator.DCSource(amplitude=3.5, start=1.0, stop=400.0) # Create the current
#currentt.inject_into(retinal_neurons) # Inject the current

# Record the voltage
retinal_neurons.record('v')
lgn_neurons.record(['v', 'spikes'])

# Run the simulation
simulator.run(t)  # Run the simulations for t ms
simulator.end()

# Extract the data
retinal_data = retinal_neurons.get_data()
retinal_segments = retinal_data.segments[0]
retinal_array = retinal_segments.analogsignalarrays[0]

lgn_data = lgn_neurons.get_data()
lgn_segments = lgn_data.segments[0]
lgn_array = lgn_segments.analogsignalarrays[0]

# Get spikes
lgn_spikes = lgn_segments.spiketrains[0] # Get the spikes
y = np.ones_like(lgn_spikes)

spike_times = np.zeros(lgn_spikes.size)
for i in xrange(spike_times.size):
    spike_times[i] = lgn_spikes[i] * 10

spike_times = spike_times.astype(int)  # Make it index

# Calculate STA
window_size = 100
sta = np.zeros(window_size)

aux = retinal_array[:,1]

for spike in spike_times:
    sta += aux[spike:spike + window_size]

sta = sta / spike_times.size
# Get times
times = retinal_array.times

# Lgn
vm_lgn = lgn_array[:, 0]

##
# Plotting
##

# Plot the voltage traces of the retinal cells
number_to_plot = 8
plot_numbers = np.random.choice(range(N_retina), number_to_plot, replace=False)

plt.subplot(2, 3, (1, 2))
for i in plot_numbers:
    plt.plot(times, retinal_array[:, i])
    plt.hold('on')

plt.xlabel('time')
plt.ylabel('voltage')

# Plot the STA
plt.subplot(2, 3, 3)
plt.plot(sta)

# Plot the voltage trace at the LGN
plt.subplot(2, 3, 4)
plt.plot(times, vm_lgn)

plt.xlabel('time')
plt.ylabel('voltage')

# Plot the spikes the LGN
plt.subplot(2, 3, 5)

plt.plot(lgn_spikes, y*10, '*')
plt.xlabel('spike time')
plt.ylabel('Spike')



plt.show()
