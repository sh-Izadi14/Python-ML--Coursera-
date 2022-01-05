import numpy as np
from nnCostFunction import nnCostFunction

def computeNumericalGradient(nn_params, input_layer_size, hidden_layer_size,\
                                    num_labels, X, y, Lambda):
#COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
#and gives us a numerical estimate of the gradient.
#   numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
#   gradient of the function J around theta. Calling y = J(theta) should
#   return the function value at theta.

# Notes: The following code implements numerical gradient checking, and 
#        returns the numerical gradient.It sets numgrad(i) to (a numerical 
#        approximation of) the partial derivative of J with respect to the 
#        i-th input argument, evaluated at theta. (i.e., numgrad(i) should 
#        be the (approximately) the partial derivative of J with respect 
#        to theta(i).)
#                
      theta = nn_params
      numgrad = np.zeros_like(theta)
      perturb = np.zeros_like(theta)
      e = 1e-4
      for p in range(theta.size):
            # Set perturbation vector
            perturb[p,0] = e
            loss1 = nnCostFunction(theta - perturb, input_layer_size, hidden_layer_size, \
                               num_labels, X, y, Lambda)
            loss2 = nnCostFunction(theta + perturb, input_layer_size, hidden_layer_size, \
                               num_labels, X, y, Lambda)
            # Compute Numerical Gradient
            numgrad[p,0] = (loss2 - loss1) / (2*e)
            perturb[p,0] = 0
      return numgrad