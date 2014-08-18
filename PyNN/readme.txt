This files are able to run the Troyer model of the visual system with the ability to reproduce Contrast Invariant Orientation Tunning. 

In order to run this files you first select a contrast in the scripts `produce_lgn_spikes_on_cells.py' and 'produce_lgn_spikes_off_cells.py'. 

Running this files will produce and store the spikes and the cell positions of the on and off cells in the folder './data/'. There are default
files provided in the repository with a couple of values of contrast if you want to skip this particular step. 

In order to run the complete model in the 'full_model.py' script we select a contrast for which we already have spikes in the './data/' folder. 

In the 'full_model.py' script (lines 40 to 60) we have a couple of parameters that determine which parts of the models we add:

N_lgn_layers : from 1 to 4 the number of LGN layers that we want to add. Obviously the smaller this value the faster the simulation works.

thalamo_cortical_connections :  If True create connections from the thalamus to the cortex
feed_forward_inhibition : If True add feed-forward inhibition ( i -> e )
cortical_excitatory_feedback : If True add cortical excitatory feedback (e -> e) and ( e -> i )
background_noise : If True add cortical noise
correlated_noise : Makes the noise coorelated

Furthemore we also have a couple of options to decide which data to save and which analysis to perform:

# Save
save_voltage_and_conductances : Saves the value voltage and conductances of the excitatory network
save_orientation_response : Saves the firing rate as a function of the orientaion of the cell. This is the value that we need to perform the analysis to validate the model. 

# Plot
plot_conductance_analysis : Plots the evolution of the conductances in time 
plot_spike_analysis : Produces a raster plot of the excitatory and the inhibitory network
plot_orientation_analysis : For this particular value of contrast plots the firing rate of the neuron as a function of their orientations.


After running the 'full_model.py' script with a couple of contrast values the orientation dependent reponse will be stored in the folder './output_data/'. We can run the final script 'compare_contrasts.py' in order to compare the orientation tunning for the contrasts that we have available. We provide some default values in the repository if you want to jump immediately to this step. 


***
Required Software:
PyNN '0.8beta1'
Nest version: Version 2.2.2
Neuron Release 7.3 


***
Further work and missing details of the model:
- In the model there is a variable delay whereas in our model there is a fix one that is free for the user to determine
- In the model the conductance is modelled as box that rises and falls exponentially whereas in our model we just take into 
account the exponential decay.
- We are missing the machinery to test our model against moving bars 

