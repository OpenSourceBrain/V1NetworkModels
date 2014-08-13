import numpy as np
import matplotlib.pyplot as plt


folder ='./output_data/'
orientation = '_orientation'

contrasts = [0.25, 0.35, 0.50]

for contrast in contrasts:

    filename_orientation = folder + str(contrast) + orientation + '.npy'

    data = np.load(filename_orientation)

    orientations = data[0,:]
    rates = data[1, :]

    plt.plot(orientations, rates, label=str(contrast))
    plt.xlabel('Degrees')
    plt.ylabel('Firing rate (Hz)')

plt.title('Firing Rate vs Orientation for different contrasts')
plt.legend()
plt.show()


