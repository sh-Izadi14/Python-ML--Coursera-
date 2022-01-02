from scipy.optimize import least_squares
import numpy as np
from mapFeature import mapFeature
from sigmoid import sigmoid

def model(theta, X):
    theta =theta[:,np.newaxis]
    return (X @ theta).flatten()

def fun(theta, X, y):
    return model(theta, X) - y

def jac(theta, X, y):
# Initialize some useful values
    m = len(y) # number of training examples

# You need to return the following variables correctly 
    J = 0
    grad = np.zeros(np.shape(theta))

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta
#
# Note: grad should have the same dimensions as theta
#
    z = X @ theta

    h = sigmoid(z)

    #XT = np.ndarray.transpose(X)
    XT = X.T
    grad = (1/m) * (XT @ (h-y) )


    return grad

data2 = np.loadtxt('ex2data2.txt', delimiter = ",")
x = data2[:,[0,1]]
y = data2[:,2]
degree = 6
X = mapFeature(x[:,0], x[:,1], degree)

# Initialize fitting parameters
theta0 = np.zeros(n)

res = least_squares(fun, theta0, jac=jac, args=(X, y), verbose=1)