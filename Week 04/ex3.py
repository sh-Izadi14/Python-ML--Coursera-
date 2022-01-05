## Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     lrCostFunction.py (logistic regression cost function)
#     oneVsAll.py
#     predictOneVsAll.py
#     predict.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Initialization
import numpy as np
import scipy.io
from displayData import displayData
from lrCostFunction import lrCostFunction, lrGradFunction
from oneVsAll import oneVsAll
from predictOneVsAll import predictOneVsAll

## Setup the parameters you will use for this part of the exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
num_labels = 10          # 10 labels, from 0 to 9

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...\n')

mat = scipy.io.loadmat('ex3data1.mat') # loading the data as a python dictionary
X = mat['X'] # extract X 
y = mat['y'] # extract y

index10 = np.argwhere(y == 10) # finding indices where y is equal to 10
y[index10] = 0 # changing y = 10 with y = 0 (cause python indexing start from 0, no need to map 0 to 10) 

m, n = X.shape

# Randomly select 100 data points to display

rand_indices = np.random.permutation(m)
sel = X[rand_indices[0:100],:]

displayData(sel)

print('Program paused.\n')
input("Press Enter to continue...")

## ============ Part 2a: Vectorize Logistic Regression ============
#  In this part of the exercise, you will reuse your logistic regression
#  code from the last exercise. You task here is to make sure that your
#  regularized logistic regression implementation is vectorized. After
#  that, you will implement one-vs-all classification for the handwritten
#  digit dataset.
#

# Test case for lrCostFunction
print('\nTesting lrCostFunction() with regularization')

theta_t = np.array([[-2], [-1], [1], [2]])
temp1 = np.ones((5,1))
temp2 = np.arange(1,16).reshape((3,5)).T/10
X_t = np.c_[temp1, temp2]
y_t = np.array([[1],[0],[1],[0],[1]])
lambda_t = 3
cost = lrCostFunction(theta_t, X_t, y_t, lambda_t)
grad = lrGradFunction(theta_t, X_t, y_t, lambda_t)

print('\nCost: {}\n'.format(cost))
print('Expected cost: 2.534819\n')
print('Gradients:\n')
print(' {} \n'.format(grad))
print('Expected gradients:\n')
print(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n')

print('Program paused.')
input("Press Enter to continue...")

## ============ Part 2b: One-vs-All Training ============
print('\nTraining One-vs-All Logistic Regression...\n')

Lambda = 0.1

all_theta = oneVsAll(X, y, num_labels, Lambda)

print('Program paused.')
input("Press Enter to continue...")


## ================ Part 3: Predict for One-Vs-All ================

p = predictOneVsAll(all_theta, X)

print('Train Accuracy: {0}\n'.format(np.mean((p.flatten() == y.flatten())) * 100))
