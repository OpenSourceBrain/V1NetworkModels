import numpy as np
from scipy.stats.stats import pearsonr
from kernel_functions import spatial_kernel

def gabor_probability(x, y, sigma, gamma, phi, w, theta, xc=0, yc=0):

    """
    calculate the gabor function of x and y

    Returns value of the 2D Gabor function at x, y

    sigma: Controls the decay of the exponential term
    gamma: x:y proportionality factor, elongates the pattern
    phi: Phase of the overall pattern
    w: Frequency of the pattern
    theta: Rotates the whole pattern by the angle theta
    xc, yc : Linear translation
    """

    # Translate
    x -= xc
    y -= yc

    # Rotate
    x = np.cos(theta) * x + np.sin(theta) * y
    y = -np.sin(theta) * x + np.cos(theta) * y

    # Function
    exp_part = np.exp(-(x**2 + (gamma * y)**2)/(2 * sigma**2))
    cos_part = np.cos(2 * np.pi * w * x + phi)

    return exp_part * cos_part


def calculate_correlations_to_cell(x_position, y_position, x_values, y_values,
                                   lx, dx, ly, dy, sigma_center, sigma_surround):
    """
    Calculates the correlations of the cell in x_positions, y_position to all other cells in the population

    Returns a vector with the pearson correlation coefficient between the cell and all other cells

    Parameters
    ---------------
    x_position, y_position : The cell's spatial coordinates
    x_values, y_values : The grid coordinates where all the other cells are located
    lx, ly : The extent of the receptive field space in the x and y direction respectively
    dx, dy : The resolution of the receptive field space in x and y respectively
    sigma_center, sigma_surround: The parameter of the center-surround receptive field structure


    """
    values = np.zeros(x_values.size * y_values.size)
    counter_aux = 0

    # Calculate the receptive field of the cell
    Z1 = spatial_kernel(lx, dx, ly, dy, sigma_center, sigma_surround, inverse=1, x_tra=x_position, y_tra=y_position)

    # Now we calclate the cross-correlation of this cell with each other cell in the gri
    for x_to in x_values:
        for y_to in y_values:

            # Call the receptive field of the cell
            Z2 = spatial_kernel(lx, dx, ly, dy, sigma_center, sigma_surround, inverse=1, x_tra=x_to, y_tra=y_to)

            # Calculate the correlation and sotre it
            values[counter_aux] = pearsonr(Z1.flat, Z2.flat)[0]
            counter_aux += 1
    return values