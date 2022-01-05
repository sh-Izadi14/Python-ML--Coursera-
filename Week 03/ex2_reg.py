## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.
#
#  You will need to complete the following functions in this exericse:
#
#     sigmoid.py
#     costFunction.py
#     predict.py
#     costFunctionReg.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

from plotData import plotData
from mapFeature import mapFeature
from costFunction import costFunctionReg, gradFunctionReg
from plotDecisionBoundary import plotDecisionBoundary
from predict import predict

## Load Data
#  The first two columns contains the X values and the third column
#  contains the label (y).


data2 = np.loadtxt('ex2data2.txt', delimiter = ",")
X = data2[:,[0,1]]
y = data2[:,2]
y=y[:,np.newaxis]

plotData(X, y)
# Put some labels 
# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend(loc='best', shadow=False, fontsize='small')
# Specified in plot order
plt.legend(["y = 1", "y = 0"], loc='best', \
    shadow=False, fontsize='small')
plt.autoscale(enable=True, axis='x', tight=True)
plt.autoscale(enable=True, axis='y', tight=True)

## =========== Part 1: Regularized Logistic Regression ============
#  In this part, you are given a dataset with data points that are not
#  linearly separable. However, you would still like to use logistic
#  regression to classify the data points.
#
#  To do so, you introduce more features to use -- in particular, you add
#  polynomial features to our data matrix (similar to polynomial
#  regression).
#

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
degree = 6
X = mapFeature(X[:,0], X[:,1], degree)

# Initialize fitting parameters
initial_theta = np.zeros([X.shape[1],1])

# Set regularization parameter Lambda to 1
Lambda = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
cost = costFunctionReg(initial_theta, X, y, Lambda)
grad = gradFunctionReg(initial_theta, X, y, Lambda)

print('Cost at initial theta (zeros): {0}'.format(cost))
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros) - first five values only:')
print(' {0} \n'.format(grad[0:5]))
print('Expected gradients (approx) - first five values only:')
print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')

print('\nProgram paused. Press enter to continue.\n')
input("Press Enter to continue...")

# Compute and display cost and gradient
# with all-ones theta and Lambda = 10
Lambda = 10
test_theta = np.ones([X.shape[1],1])
cost = costFunctionReg(test_theta, X, y, Lambda)
grad = gradFunctionReg(test_theta, X, y, Lambda)

print('\nCost at test theta (with Lambda = 10): {0}'.format(cost))
print('Expected cost (approx): 3.16\n')
print('Gradient at test theta - first five values only:\n')
print(' {0} \n'.format(grad[0:5]))
print('Expected gradients (approx) - first five values only:\n');
print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n');

print('\nProgram paused. Press enter to continue.\n')
input("Press Enter to continue...")

## ============= Part 2: Regularization and Accuracies =============
#  Optional Exercise:
#  In this part, you will get to try different values of Lambda and
#  see how regularization affects the decision coundart
#
#  Try the following values of Lambda (0, 1, 10, 100).
#
#  How does the decision boundary change when you vary Lambda? How does
#  the training set accuracy vary?
#

# Initialize fitting parameters
initial_theta = np.zeros([X.shape[1],1])

# Set regularization parameter Lambda to 1 (you should vary this)
Lambda = 1

# fminunc
temp = opt.fmin_tnc(func = costFunctionReg, 
                    x0 = initial_theta.flatten(),fprime = gradFunctionReg, 
                    args = (X, y.flatten(), Lambda))

theta_optimized = temp[0]
theta_optimized = theta_optimized.reshape(([X.shape[1],1]))
print(theta_optimized)
# Plot Boundary
plotDecisionBoundary(theta_optimized, X, y, degree = degree)
plt.legend(["y = 1", "y = 0"], loc='best', shadow=False, fontsize='small')
# title(sprintf('Lambda = #g', Lambda))

# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')


# Compute accuracy on our training set
p = predict(theta_optimized, X)

print('Train Accuracy: {0}\n'.format(np.mean((p.flatten() == y.flatten())) * 100))
print('Expected accuracy (with Lambda = 1): 83.1 (approx)')


plt.show()