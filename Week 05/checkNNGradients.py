import numpy as np
from debugInitializeWeights import debugInitializeWeights
from nnCostFunction import nnGradFunction
from computeNumericalGradient import computeNumericalGradient

def checkNNGradients(Lambda):
      
      input_layer_size = 3
      hidden_layer_size = 5
      num_labels = 3
      m = 5

      # We generate some 'random' test data
      Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
      Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)
      # Reusing debugInitializeWeights to generate X
      X  = debugInitializeWeights(m, input_layer_size - 1)
      y  = 1 + (np.arange(0,m,1,dtype=int) % num_labels).reshape((m,1))

      # Unroll parameters
      nn_params = Theta1.reshape((np.size(Theta1),1))
      nn_params = np.vstack((nn_params,Theta2.reshape(Theta2.size,1)))

      # Short hand for cost function
      grad = nnGradFunction(nn_params, input_layer_size, hidden_layer_size,\
                                    num_labels, X, y, Lambda)
      
      numgrad = computeNumericalGradient(nn_params, input_layer_size, hidden_layer_size,\
                                    num_labels, X, y, Lambda)

      # Visually examine the two gradient computations.  The two columns
      # you get should be very similar. 
      print(np.c_[numgrad, grad])
      print('The above two columns you get should be very similar.\n', \
            '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n')

      # Evaluate the norm of the difference between two solutions.  
      # If you have a correct implementation, and assuming you used EPSILON = 0.0001 
      # in computeNumericalGradient.m, then diff below should be less than 1e-9
      diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)

      print('If your backpropagation implementation is correct, then \n', \
            'the relative difference will be small (less than 1e-9). \n', \
            '\nRelative Difference: {}\n'.format(diff))