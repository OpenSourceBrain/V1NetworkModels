__author__ = 'heberto'


import pyNN.nest as simulator
import pyNN.nest.standardmodels.electrodes as elect

N = 1  # Number of neurons
t = 100.0  #Simulation time

# Has to be called at the beginning of the simulation
simulator.setup(timestep=0.1, min_delay=0.1, max_delay=10)

model = simulator.IF_curr_exp()

neurons = simulator.Population(N, model)

# DC source
current = simulator.DCSource(amplitude=0.5, start=20.0, stop=80.0)
#current = elect.DCSource(amplitude=0.5, start=20.0, stop=80.0)
current.inject_into(neurons)
#neurons.inject(current)

# Record the voltage
neurons.record('v')

simulator.run(t)  # Run the simulations for t ms

simulator.end()

# Extracts the data
data = neurons.get_data() # Crates a Neo Block
segment = data.segments[0] # Takes the first
vm = segment.analogsignalarrays[0]

# Plot the data
import matplotlib.pyplot as plt
plt.plot(vm.times, vm)
plt.xlabel('time')
plt.ylabel('Vm')
plt.show()