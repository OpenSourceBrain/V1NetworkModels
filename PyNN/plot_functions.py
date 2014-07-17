import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def visualize_firing_rate(firing_rate, dt, T_simulation, label=None):
    """
    Plot the firing rate 
    """
    
    t = np.arange(0, T_simulation, dt)
    plt.plot(t, firing_rate, label=label)
    plt.xlabel('Time (ms)')
    plt.ylabel('Firing rate (Hz)')
    plt.legend()
    


def plot_mutliplot_bilinear(N,ims, colorbar=True, symmetric=0):
    """
    Plots a series of 2D images as imshow functions. 
    
    ---
    Parameters
    N: This is the number of actual images that are going to be ploted. This function will round to the closest perfect square 
    number n and then it will make an array of n x n figures.
    
    colorbar: True if you want the colorbars and False if the oposite
    
    symmetric:  0 sets a symmetric ticks with maximum of abs(ims) as vmax and the negative of that value a vmin
                1 set the maximum of ims as vmax and minimum as min
                2 no values, default of imshow
    """
    
    # Set the color map
    cdict1 = {'red':   ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.1),
                   (1.0, 1.0, 1.0)),
    
         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
    
         'blue':  ((0.0, 0.0, 1.0),
                   (0.5, 0.1, 0.0),
                   (1.0, 0.0, 0.0))
        }
    
    blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1)
    
    # Set the maximums and minimums 
    if symmetric == 0:
        vmax = np.max( (np.abs(np.min(ims)), np.max(ims))) 
        vmin = - vmax
    elif symmetric ==1 :
        vmax = np.max(ims)
        vmin = np.min(ims)
    else:
        vmax = None
        vmin = None
    
    n = int(np.sqrt(N))
    print n
    number = n * n
    print number
    for i in range(number):
        plt.subplot(n,n,i + 1)
        plt.imshow(ims[i,:,:], interpolation='bilinear', cmap=blue_red1, vmin=vmin, vmax=vmax)
        if colorbar==True:
            plt.colorbar()

def plot_spiketrains(segment):
    """
    Plots the spikes of all the cells in the given segments 
    """
    for spiketrain in segment.spiketrains:
        y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
        plt.plot(spiketrain, y, '*b')
    plt.ylabel('Neuron number')
    plt.xlabel('Spikes')
