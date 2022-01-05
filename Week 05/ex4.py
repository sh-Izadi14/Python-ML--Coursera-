## Machine Learning Online Class - Exercise 4 Neural Network Learning

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     sigmoidGradient.m
#     randInitializeWeights.m
#     nnCostFunction.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Initialization
import numpy as np
import scipy.io
import scipy.optimize as opt

from displayData import displayData
from predict import predict
from nnCostFunction import nnCostFunction, nnGradFunction
from randInitializeWeights import randInitializeWeights
from sigmoidGradient import sigmoidGradient
from checkNNGradients import checkNNGradients


## Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   
                          # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...\n')

mat = scipy.io.loadmat('ex4data1.mat') # loading the data as a python dictionary
X = mat['X'] # extract X 
y = mat['y'] # extract y 

m, n = X.shape

# Randomly select 100 data points to display

rand_indices = np.random.permutation(m)
sel = X[rand_indices[0:100],:]

displayData(sel)

print('Program paused.\n')
input("Press Enter to continue...")


## ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print('\nLoading Saved Neural Network Parameters ...\n')

# Load the weights into variables Theta1 and Theta2
mat2 = scipy.io.loadmat('ex4weights.mat')
Theta1 = mat2['Theta1']
Theta2 = mat2['Theta2']

# Unroll parameters 
nn_params = Theta1.reshape((np.size(Theta1),1))
nn_params = np.vstack((nn_params,Theta2.reshape(Theta2.size,1)))
## ================ Part 3: Compute Cost (Feedforward) ================
#  To the neural network, you should first start by implementing the
#  feedforward part of the neural network that returns the cost only. You
#  should complete the code in nnCostFunction.m to return cost. After
#  implementing the feedforward to compute the cost, you can verify that
#  your implementation is correct by verifying that you get the same cost
#  as us for the fixed debugging parameters.
#
#  We suggest implementing the feedforward cost *without* regularization
#  first so that it will be easier for you to debug. Later, in part 4, you
#  will get to implement the regularized cost.
#
print('\nFeedforward Using Neural Network ...\n')

# Weight regularization parameter (we set this to 0 here).
Lambda = 0

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, \
                   num_labels, X, y, Lambda)

print('Cost at parameters (loaded from ex4weights): {} '.format(J), \
      '\n(this value should be about 0.287629)\n')

print('Program paused.')
input('Press enter to continue...')

## =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.
#

print('\nChecking Cost Function (w/ Regularization) ... \n')

# Weight regularization parameter (we set this to 1 here).
Lambda = 1

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, \
                   num_labels, X, y, Lambda)

print('Cost at parameters (loaded from ex4weights): {} '.format(J), \
         '\n(this value should be about 0.383770)')

print('Program paused.')
input('Press enter to continue...')

## ================ Part 5: Sigmoid Gradient  ================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.m file.
#

print('\nEvaluating sigmoid gradient...\n')

g = sigmoidGradient(np.array([[-1, -0.5, 0, 0.5, 1]]))
print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]: ')
print('{} '.format(g))
print('\n')

print('Program paused.')
input('Press enter to continue...')


## ================ Part 6: Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two
#  layer neural network that classifies digits. You will start by
#  implementing a function to initialize the weights of the neural network
#  (randInitializeWeights.m)

print('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = initial_Theta1.reshape((np.size(initial_Theta1),1))
initial_nn_params = np.vstack((initial_nn_params, \
      initial_Theta2.reshape(initial_Theta2.size,1)))

## =============== Part 7: Implement Backpropagation ===============
#  Once your cost matches up with ours, you should proceed to implement the
#  backpropagation algorithm for the neural network. You should add to the
#  code you've written in nnCostFunction.m to return the partial
#  derivatives of the parameters.
#
print('Checking Backpropagation... \n')

#  Check gradients by running checkNNGradients
checkNNGradients(Lambda = 0)

print('\nProgram paused.')
input('Press enter to continue...')


## =============== Part 8: Implement Regularization ===============
#  Once your backpropagation implementation is correct, you should now
#  continue to implement the regularization with the cost and gradient.
#

print('\nChecking Backpropagation (w/ Regularization) ... \n')

#  Check gradients by running checkNNGradients
Lambda = 3
checkNNGradients(Lambda)

# Also output the costFunction debugging values
debug_J  = nnCostFunction(nn_params, input_layer_size, \
                          hidden_layer_size, num_labels, X, y, Lambda)

print('\nCost at (fixed) debugging parameters (w/ Lambda = {}): {}'.format(Lambda, debug_J),
         '\n(for Lambda = 3, this value should be about 0.576051)\n')

print('Program paused.')
input('Press enter to continue...')


## =================== Part 8: Training NN ===================
#  You have now implemented all the code necessary to train a neural 
#  network. To train your neural network, we will now use "fmincg", which
#  is a function which works similarly to "fminunc". Recall that these
#  advanced optimizers are able to train our cost functions efficiently as
#  long as we provide them with the gradient computations.
#
print('\nTraining Neural Network... \n')

#  You should also try different values of Lambda
Lambda = 1

# res = opt.fmin_tnc(func = nnCostFunction, x0 = initial_nn_params.flatten(), \
#                    args = (input_layer_size, hidden_layer_size, num_labels, \
#                          X, y.flatten(), Lambda), fprime = nnGradFunction, messages = 5)

res = opt.minimize(fun = nnCostFunction, x0 = initial_nn_params.flatten(), \
                   args = (input_layer_size, hidden_layer_size, num_labels, \
                         X, y.flatten(), Lambda), method='TNC', jac = nnGradFunction, options={'disp':True, 'maxiter': 150})

# Obtain Theta1 and Theta2 back from nn_params
nn_params = res.x
Theta1 = nn_params[:((input_layer_size+1) * hidden_layer_size)].reshape(hidden_layer_size,input_layer_size+1)

Theta2 = nn_params[((input_layer_size +1)* hidden_layer_size ):].reshape(num_labels,hidden_layer_size+1)

print('Program paused.')
input('Press enter to continue...')


## ================= Part 9: Visualize Weights =================
#  You can now "visualize" what the neural network is learning by 
#  displaying the hidden units to see what features they are capturing in 
#  the data.

print('\nVisualizing Neural Network... \n')

displayData(Theta1[:,1:])

print('Program paused.')
input('Press enter to continue...')

## ================= Part 10: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X)

print('Train Accuracy: {0}\n'.format(np.mean((pred.flatten() == y.flatten())) * 100))
