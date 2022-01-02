import numpy as np
from sigmoid import sigmoid

def costFunction(theta, X, y):
    #COSTFUNCTION Compute cost and gradient for logistic regression
#   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
#   parameter for logistic regression and the gradient of the cost
#   w.r.t. to the parameters.

# Initialize some useful values
    m = len(y) # number of training examples

# You need to return the following variables correctly 
    J = 0
    grad = np.zeros_like(theta)

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta
#
# Note: grad should have the same dimensions as theta
#
    z = np.dot(X, theta)

    h = sigmoid(z)

    #hT = np.ndarray.transpose(h)
    J = (-1/m)* ( np.dot( np.log(h.T), y) + np.dot(np.log(1- h.T), (1-y)))

    return J

def gradFunction(theta, X, y):

# You need to return the following variables correctly 
    grad = np.zeros_like(theta)

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta
#
# Note: grad should have the same dimensions as theta
#

    m = len(y)
    z = np.dot(X, theta)
    h = sigmoid(z)
    grad = (1/m) * np.dot(X.T, (h-y) )


    return grad

# =============================================================
def costFunctionReg(theta, X, y, Lambda):
#COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
#   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
#   theta as the parameter for regularized logistic regression and the
#   gradient of the cost w.r.t. to the parameters. 

# Initialize some useful values
    m = len(y); # number of training examples


    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
  
    z = X @ theta
    h = sigmoid(z)

    theta = np.delete(theta, 0)
    theta = theta.reshape(len(theta),1)

    X     = np.delete(X,0,1)

    J = (-1/m) * ( np.log(h.T) @ y + np.log(1-h.T) @ (1-y) ) + (Lambda/(2*m)) *\
       (theta.T @ theta)
    
    return J

def gradFunctionReg(theta, X, y, Lambda):

    m = len(y)
    grad = np.zeros([m,1])
    grad = (1/m) * X.T @ (sigmoid(X @ theta) - y)
    grad[1:] = grad[1:] + (Lambda / m) * theta[1:]
    
    return grad
    