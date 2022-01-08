## Machine Learning Online Class
#  Exercise 5 | Regularized Linear Regression and Bias-Variance
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     linearRegCostFunction.m
#     learningCurve.m
#     validationCurve.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Initialization
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from linearRegCostFunction import linearRegCostFunction, linearRegGradientFunction
from trainLinearReg import trainLinearReg
from learningCurve import learningCurve
from polyFeatures import polyFeatures
from featureNormalize import featureNormalize
from plotFit import plotFit
from validationCurve import validationCurve

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.
#

# Load Training Data
print('Loading and Visualizing Data ...\n')

# Load from ex5data1: 
# You will have X, y, Xval, yval, Xtest, ytest in your environment
mat = scipy.io.loadmat('ex5data1.mat')
X = mat['X']
y = mat['y']

Xval = mat['Xval']
yval = mat['yval']

Xtest = mat['Xtest']
ytest = mat['ytest']
# m = Number of examples
m = X.shape[0]

# Plot training data
plt.plot(X, y, 'rx', markersize =10)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')

plt.show()

print('Program paused.')
input('Press enter to continue...')

## =========== Part 2: Regularized Linear Regression Cost =============
#  You should now implement the cost function for regularized linear 
#  regression. 
#

theta = np.array([[1], [1]])
J = linearRegCostFunction(theta, np.c_[np.ones((m, 1)), X], y, 1)

print('Cost at theta = [1  1]: {}'.format(J), \
        '\n(this value should be about 303.993192)')

print('Program paused.')
input('Press enter to continue...')

# =========== Part 3: Regularized Linear Regression Gradient =============
#  You should now implement the gradient for regularized linear 
#  regression.
#

theta = np.array([[1], [1]])
grad = linearRegGradientFunction(theta, np.c_[np.ones((m, 1)), X], y, 1)

print('Gradient at theta = [1  1]:  {} '.format(grad.T), \
         '\n(this value should be about [-15.303016 598.250744])\n')

print('Program paused.')
input('Press enter to continue...')

## =========== Part 4: Train Linear Regression =============
#  Once you have implemented the cost and gradient correctly, the
#  trainLinearReg function will use your cost function to train 
#  regularized linear regression.
# 
#  Write Up Note: The data is non-linear, so this will not give a great 
#                 fit.
#

#  Train linear regression with lambda = 0
Lambda = 0
theta = trainLinearReg(np.c_[np.ones((m, 1)), X], y, Lambda)

#  Plot fit over the data
plt.plot(X, y, 'rx')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')

plt.plot(X, np.c_[np.ones((m, 1)), X]@theta, '--')

plt.show()

print('Program paused.')
input('Press enter to continue...')

## =========== Part 5: Learning Curve for Linear Regression =============
#  Next, you should implement the learningCurve function. 
#
#  Write Up Note: Since the model is underfitting the data, we expect to
#                 see a graph with "high bias" -- Figure 3 in ex5.pdf 
#

Lambda = 0

error_train, error_val = learningCurve(np.c_[np.ones((m, 1)), X], y,\
         np.c_[np.ones((Xval.shape[0],1)), Xval], yval, Lambda)

plt.plot(np.arange(1,m+1), error_train, label= 'Train')
plt.plot(np.arange(1,m+1), error_val, label= 'Cross Validation')
plt.title('Learning curve for linear regression')
plt.legend()
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 150])

plt.show()

print('# Training Examples\tTrain Error\tCross Validation Error\n')

for i in range(m):
    print('\t{}\t\t{}\t{}\n'.format(i, error_train[i,0], error_val[i,0]))

print('Program paused.')
input('Press enter to continue...')

## =========== Part 6: Feature Mapping for Polynomial Regression =============
#  One solution to this is to use polynomial regression. You should now
#  complete polyFeatures to map each example into its powers
#

p = 8

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p)
X_poly, mu, sigma = featureNormalize(X_poly)  # Normalize
X_poly = np.c_[np.ones((m, 1)), X_poly]                   # Add Ones

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test = X_poly_test - mu
X_poly_test = X_poly_test / sigma
X_poly_test = np.c_[np.ones((X_poly_test.shape[0], 1)), X_poly_test]         # Add Ones

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p)
X_poly_val = X_poly_val - mu
X_poly_val = X_poly_val / sigma
X_poly_val = np.c_[np.ones((X_poly_val.shape[0], 1)), X_poly_val]            # Add Ones

print('Normalized Training Example 1: ')
print('  {}  \n'.format(X_poly[0, :]))

print('Program paused.')
input('Press enter to continue...')

## =========== Part 7: Learning Curve for Polynomial Regression =============
#  Now, you will get to experiment with polynomial regression with multiple
#  values of lambda. The code below runs polynomial regression with 
#  lambda = 0. You should try running the code with different values of
#  lambda to see how the fit and learning curve change.
#

Lambda = 1
theta = trainLinearReg(X_poly, y, Lambda)

# Plot training data and fit
plt.figure(1)
plt.plot(X, y, 'rx')
plotFit(np.min(X), np.max(X), mu, sigma, theta, p)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title('Polynomial Regression Learning Curve (lambda = {})'.format(Lambda))


plt.figure(2)
error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, Lambda)

plt.plot(np.arange(1,m+1), error_train, label= 'Train')
plt.plot(np.arange(1,m+1), error_val, label= 'Cross Validation')

plt.title('Polynomial Regression Learning Curve (lambda = {})'.format(Lambda))
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 100])
plt.legend()

plt.show()

print('Polynomial Regression (lambda = {})\n\n'.format(Lambda))
print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(m):
    print('  \t{}\t\t{}\t{}\n'.format(i, error_train[i,0], error_val[i,0]))

print('Program paused.')
input('Press enter to continue...')

## =========== Part 8: Validation for Selecting Lambda =============
#  You will now implement validationCurve to test various values of 
#  lambda on a validation set. You will then use this to select the
#  "best" lambda value.
#

Lambda_vec, error_train, error_val = \
    validationCurve(X_poly, y, X_poly_val, yval)

plt.plot(Lambda_vec, error_train, label = 'Train')
plt.plot(Lambda_vec, error_val, label = 'Cross Validation')

plt.legend()
plt.xlabel('lambda')
plt.ylabel('Error')

plt.show()

print('lambda\t\tTrain Error\tValidation Error\n')
for i in range(len(Lambda_vec)):
        print(' {}\t{}\t{}\n'.format(Lambda_vec[i], error_train[i], error_val[i]))
           
