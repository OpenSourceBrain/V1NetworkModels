import numpy as np
import matplotlib.pyplot as plt
import cPickle
import pyNN.nest as simulator

contrast = 0.50
Nside_lgn = 30
Ncell_lgn = Nside_lgn * Nside_lgn
N_lgn_layers = 1
t = 1000  # ms

simulator.setup(timestep=0.1, min_delay=0.1, max_delay=5.0)

directory = './data/'
format = '.cpickle'

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


# Now we construct the functions that will pass the spikes to the cell model


def spike_times(simulator, layer, spikes_file):
    return [simulator.Sequence(x) for x in spikes_file[layer]]


# Cells models for the LGN spikes (SpikeSourceArray)
lgn_spikes_on_models = []
lgn_spikes_off_models = []


for layer in xrange(N_lgn_layers):
    model = simulator.SpikeSourceArray(spike_times=spike_times(simulator, layer, spikes_on))
    lgn_spikes_on_models.append(model)
    model = simulator.SpikeSourceArray(spike_times=spike_times(simulator, layer, spikes_off))
    lgn_spikes_off_models.append(model)

# LGN Popluations

lgn_on_populations = []
lgn_off_populations = []

for layer in xrange(N_lgn_layers):
    population = simulator.Population(Ncell_lgn, lgn_spikes_on_models[layer], label='LGN_on_layer_' + str(layer))
    lgn_on_populations.append(population)
    population = simulator.Population(Ncell_lgn, lgn_spikes_off_models[layer], label='LGN_off_layer_' + str(layer))
    lgn_off_populations.append(population)

#############################
# Recordings
#############################

layer = 0

population_on = lgn_on_populations[layer]
population_off = lgn_off_populations[layer]

population_on.record('spikes')
population_off.record('spikes')

#############################
# Run model
#############################

simulator.run(t)  # Run the simulations for t ms
simulator.end()

#############################
# Extract the data
#############################
data_on = population_on.get_data()  # Creates a Neo Block
data_off = population_off.get_data()

segment_on = data_on.segments[0]  # Takes the first segment
segment_off = data_off.segments[0]


# Plot spike trains
def plot_spiketrains(segment):
    """
    Plots the spikes of all the cells in the given segments
    """
    for spiketrain in segment.spiketrains:
        y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
        plt.plot(spiketrain, y, '*b')
    plt.ylabel('Neuron number')
    plt.xlabel('Spikes')


plt.subplot(2, 1, 1)
plt.title('On cells ')
plot_spiketrains(segment_on)

plt.subplot(2, 1, 2)
plt.title('Off cells ')
plot_spiketrains(segment_off)

plt.show()

