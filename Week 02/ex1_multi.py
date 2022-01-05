## Machine Learning Online Class
#  Exercise 1: Linear regression with multiple variables
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear regression exercise. 
#
#  You will need to complete the following functions in this 
#  exericse:
#
#     warmUpExercise.py
#     plotDatapy
#     gradientDescent.py
#     computeCost.py
#     gradientDescentMulti.py ---> (It's the same as ComputeCost.py)
#     computeCostMulti.py ---> (It's the same as ComputeCost.py)
#     featureNormalize.py
#     normalEqn.py
#
#  For this part of the exercise, you will need to change some
#  parts of the code below for various experiments (e.g., changing
#  learning rates).
#

## Initialization
import numpy as np
import matplotlib.pyplot as plt
from gradientDescent import batchGD
from featuresNormalization import featureNormalization
from normalEquation import normalEquation

## ================ Part 1: Feature Normalization ================

print('Loading data ...')

## Load Data
data2 = np.loadtxt('ex1data2.txt', delimiter = ",")
x = data2[:,[0,1]]
y = data2[:,2]
m = len(y)
y = y.reshape((m, 1))

# Print out some data points
print('\nFirst 10 examples from the dataset: \n')
for i in range(10):
    print(' x = {0}, y = {1}'.format(x[i,:], y[i]))

input("\nPress Enter to continue...")

# Scale features and set them to zero mean
print('\nNormalizing Features ...')

#Temp, mu, sigma = featureNormalize(X)
X, mu, sigma = featureNormalization(x)
print('Features Normalized')

# Add intercept term to X
temp = np.ones((m,1))
X = np.c_[temp,X] 
input("\nPress Enter to continue...")

## ================ Part 2: Gradient Descent ================

# ====================== YOUR CODE HERE ======================
# Instructions: We have provided you with the following starter
#               code that runs gradient descent with a particular
#               learning rate (alpha). 
#
#               Your task is to first make sure that your functions - 
#               computeCost and gradientDescent already work with 
#               this starter code and support multiple variables.
#
#               After that, try running gradient descent with 
#               different values of alpha and see which one gives
#               you the best result.
#
#               Finally, you should complete the code at the end
#               to predict the price of a 1650 sq-ft, 3 br house.
#
# Hint: By using the 'hold on' command, you can plot multiple
#       graphs on the same figure.
#
# Hint: At prediction, make sure you do the same feature normalization.
#

print('\nRunning gradient descent ...')

# Choose some alpha value
alpha =  0.3
num_iters = 400

# Init Theta and Run Gradient Descent 
theta = np.zeros((3, 1))

theta, J_history, iter = batchGD(X, y, theta, alpha, num_iters)

# Plot the convergence graph
plt.figure(1)
plt.plot(range(1,len(J_history)+1), J_history, '-b', linewidth = 2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')

# Display gradient descent's result
print('Theta computed from gradient descent: ')
print(theta)

input("\nPress Enter to continue...")

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.

NewData = (np.array([1650, 3])- mu)/sigma
NewData = np.insert(NewData, 0, 1) 
price = NewData @ theta # You should change this

# ============================================================

print('\nPredicted price of a 1650 sq-ft, 3 br house \
(using gradient descent):\n {0}'.format(price));

input("\nPress Enter to continue...")

## ================ Part 3: Normal Equations ================
print('\nSolving with normal equations...');

# ====================== YOUR CODE HERE ======================
# Instructions: The following code computes the closed form 
#               solution for linear regression using the normal
#               equations. You should complete the code in 
#               normalEqn.m
#
#               After doing so, you should complete this code 
#               to predict the price of a 1650 sq-ft, 3 br house.
#

temp = np.ones((m,1))
X = np.c_[temp,x]  # Add a column of ones to x

# Calculate the parameters from the normal equation
theta = normalEquation(X, y)

# Display normal equation's result
print('\nTheta computed from the normal equations: ');
print(theta)

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
NewData = np.array([1, 1650 , 3]) 
price = NewData @ theta # You should change this

# ============================================================

print('\nPredicted price of a 1650 sq-ft, 3 br house \
(using normal equations): \n{0}'.format(price))

plt.show()
