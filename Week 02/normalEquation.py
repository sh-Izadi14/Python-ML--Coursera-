import numpy as np
from numpy.linalg import pinv

def normalEquation(X, y):

    return pinv(X.T @ X) @ (X.T @ y)
