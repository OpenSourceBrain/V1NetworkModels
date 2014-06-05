__author__ = 'heberto'

import pyNN.nest as simulator
import pyNN.nest.standardmodels.electrodes as elect
import matplotlib.pyplot as plt
import numpy as np

N = 3  # Number of neurons
t = 150.0  # Simulation time

# Has to be called at the beginning of the simulation
simulator.setup(timestep=0.1, min_delay=0.1, max_delay=10)

# Neuron Model's parameters
i_offset = 0
R = 20
tau_m = 20.0
tau_refractory = 50
v_thresh = 0
v_rest = -60
tau_syn_E = 5.0
tau_syn_I = 5.0
cm = tau_m / R

# Declare our cell model
model = simulator.IF_curr_exp(cm=cm, i_offset=i_offset, tau_m=tau_m, tau_refrac=tau_refractory, tau_syn_E=tau_syn_E,
                           tau_syn_I=tau_syn_I, v_reset=v_rest, v_thresh=v_thresh)

neurons = simulator.Population(N, model)

# Create views
neuron1 = neurons[[0]]
neuron2 = neurons[1, 2]

# Modify second neuron
tau_m2 = 10.0
cm2 = tau_m2 / R
neurons[1].set_parameters(cm=cm2, tau_m=tau_m2)

tau_m3 = 15.0
cm3 = tau_m3 / R
neurons[2].set_parameters(cm=cm3, tau_m=tau_m3)

# Synapses
syn = simulator.StaticSynapse(weight=10, delay=0.5)
# Projections
connections = simulator.Projection(neuron1, neuron2, simulator.AllToAllConnector(),
                                   syn, receptor_type='excitatory')

# DC source
current = elect.DCSource(amplitude=3.5, start=20.0, stop=100.0)
current.inject_into(neuron1)
#neurons.inject(current)

# Record the voltage
neurons.record('v')

simulator.run(t)  # Run the simulations for t ms
simulator.end()

# Extracts the data
data = neurons.get_data()  # Creates a Neo Block
segment = data.segments[0]  # Takes the first segment
vm = segment.analogsignalarrays[0]  # Take the arrays

# Extract the data for neuron 1
vm1 = vm[:, 0]
vm2 = vm[:, 1]
vm3 = vm[:, 2]

# Plot the data
plt.plot(vm.times, vm1, label='pre-neuron')
plt.hold('on')
plt.plot(vm.times, vm2, label='post-neuron 1')
plt.plot(vm.times, vm3, label= 'post-neuron 2')

plt.xlabel('time')
plt.ylabel('Voltage')
plt.legend()

plt.show()