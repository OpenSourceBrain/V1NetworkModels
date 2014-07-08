import numpy as np


def gabor_probability(x, y, sigma, gamma, phi, w, theta, xc=0, yc=0):
    '''
    calculate the gabor function of x and y
    '''


    x = np.cos(theta) * x + np.sin(theta) * y
    y = -np.sin(theta) * x + np.cos(theta) * y

    exp_part = np.exp(-(x**2 + (gamma * y)**2)/(2 * sigma**2))
    cos_part = np.cos(2 * np.pi * w * x + phi)

    return  exp_part * cos_part