__author__ = 'heberto'


import numpy as np
import matplotlib.pyplot as plt

start = 30
x = np.arange(-start, start, 0.01)


alpha = 20
beta = 0.05
sin = np.sin(beta * 2 * np.pi * x)
exp = np.exp(-(x / alpha) ** 2)

y = exp * sin

plt.plot(x, sin, label='sine')
plt.hold('on')
plt.plot(x, exp, label='exponential')
plt.plot(x, y, label='gabor')
plt.grid()
plt.legend()
plt.show()