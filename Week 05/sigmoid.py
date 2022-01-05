import numpy as np

# sigmoid Function
def sigmoid(z):
    g = np.zeros(np.shape(z))
    g = 1/(1 + np.exp(-z))
    return g