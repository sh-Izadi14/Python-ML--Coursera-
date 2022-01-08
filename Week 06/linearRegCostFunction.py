import numpy as np

def linearRegCostFunction(theta, X, y, Lambda):
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
  
    h = X @ theta

    J = (1/(2*m)) * ( h - y).T @ ( h - y) + (Lambda/(2*m)) * sum(theta[1:]**2)
    
    return J

def linearRegGradientFunction(theta, X, y, Lambda):

    m = len(y)
    grad = np.zeros_like(theta)
    grad = (1/m) * X.T @ ((X @ theta) - y)
    grad[1:] = grad[1:] + (Lambda / m) * theta[1:]
    
    return grad
    