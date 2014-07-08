import numpy as np
import matplotlib.pyplot as plt 


def gabor_kernel(lx, dx, ly, dy, sigma, gamma, phi, w, theta, xc=0, yc=0):
    '''
    Produces a gabor pattern. That is, the product of an exponential
    term and a sinusoidal term 
    
    Parameters
    -----------
    lx, ly : Wide of the spatial kernel in x and y respectively 
    dx, dy : Resolution in x and y respectively 
    sigma: Controls the decay of the exponential term
    gamma: x:y proportinality factor, elongates the pattern
    phi: Phase of the overall pattern 
    w: Frequency of the pattern 
    theta: Rotates the whole pattern by the angle theta
    
    '''
    
    x = np.arange(-lx/2, lx/2, dx)
    y = np.arange(-ly/2, ly/2, dy)
    
    X, Y = np.meshgrid(x, y)
    
    X = np.cos(theta) * X + np.sin(theta) * Y
    Y = - np.sin(theta) * X + np.cos(theta) * Y
    
    exp_part = np.exp(-((X - xc)**2 + (gamma * (Y - yc))**2)/(2 * sigma**2))
    cos_part = np.cos(2 * np.pi * w * (X - xc) + phi)

    return exp_part * cos_part


def spatial_kernel(lx, dx, ly, dy, sigma_center, sigma_surround, inverse=1, x_tra=0, y_tra=0):
    """
    The spatial component of the kernel. A difference of gaussian'
    
    Parameters
    --------------
    lx, ly : Wide of the spatial kernel in x and y respectively 
    dx, dy : Resolution in x and y respectively 
    sigma_center: Related to the size of the center region 
    sigma_surround: Related to the size of the surround region
    inverse : if -1 inverts the sign of the whole pattern.  That is, 
                it switches from on-off to off-on 
    x_tra, y_tra : linear translation of the x and y coordinates respectively
    
    """
    # Create the positions
    x = np.arange(-lx/2, lx/2, dx)
    y = np.arange(-ly/2, ly/2, dy)
    # Create a x,y plane 
    X, Y = np.meshgrid(x, y)
    # Calculate function 
    R = np.sqrt((X - x_tra)**2 + (Y - y_tra)**2)  
    center = (17.0 / sigma_center**2) * np.exp(-(R / sigma_center)**2)
    surround = (16.0 / sigma_surround**2) * np.exp(-(R / sigma_surround)**2)
    Z = center - surround
    return Z * inverse


def temporal_kernel(t):
    """
    Calculate a biphasic temporal kernel. Cite KAI
    """

    ## Temporal parameters
    K1 = 1.05
    K2 = 0.7
    c1 = 0.14
    c2 = 0.12
    n1 = 7.0
    n2 = 8.0
    t1 = -6.0
    t2 = -6.0
    td = 6.0

    #  Calculate functions 
    p1 = K1 * ((c1*(t - t1))**n1 * np.exp(-c1*(t - t1))) / ((n1**n1) * np.exp(-n1))
    p2 = K2 * ((c2*(t - t2))**n2 * np.exp(-c2*(t - t2))) / ((n2**n2) * np.exp(-n2))
    p3 = p1 - p2

    return p3


def create_kernel(dx, lx, dy, ly, sigma_surround, sigma_center, dt_kernel, kernel_size, inverse=1, x_tra=0, y_tra=0):
    """
    Returns the kernel

    Note:
    At the end we multiply the whole kernel by a scale factor. Is necessary to take into account that the bigger
    the kernel_size the more terms will be used for the convolution calculation. It is necessary then to normalize
    by the size. Furthermore, if dt_kernel is small then more terms will be include in the calculation, and taking
    this into account we also multiply by dt_kernel

    The bigger the number of kernel points (dx/ lx) or (dy/ ly) the more terms terms

    """

    # Call the spatial kernel
    Z = spatial_kernel(lx, dx, ly, dy, sigma_center, sigma_surround, inverse=inverse, x_tra=x_tra, y_tra=y_tra)
    # Call the temporal kernel
    t = np.arange(0, kernel_size * dt_kernel, dt_kernel)
    f_t = temporal_kernel(t)

    # Initialize and fill the spatio-temporal kernel  
    kernel = np.zeros((kernel_size, int(lx/dx), int(ly/dy)))

    for k, ft in enumerate(f_t):
        kernel[k,...] = ft * Z 
    
    
    return kernel * (dt_kernel) *  ( (dx/lx) * (dy/ly) )