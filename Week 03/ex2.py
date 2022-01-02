## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the logistic
#  regression exercise. You will need to complete the following functions 
#  in this exericse:
#
#     Sigmoid.py
#     costFunction.py
#     predict.py
#     costFunctionReg.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# Initialization
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

from plotData import plotData
from costFunction import costFunction, gradFunction
from sigmoid import sigmoid
from predict import predict
from plotDecisionBoundary import plotDecisionBoundary

## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.


data1 = np.loadtxt('ex2data1.txt', delimiter = ",")
X = data1[:,[0,1]]
y = data1[:,2]
m = len(y)
y=y.reshape((m,1))

## ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the 
#  the problem we are working with.

print('Plotting data with + indicating (y = 1) examples and o ' \
         'indicating (y = 0) examples.\n')

plotData(X, y)

# Put some labels 
# Labels and Legend
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(["Admitted", "Not Admitted"], loc='best', shadow=False, fontsize='small')
# Specified in plot order
plt.autoscale(enable=True, axis='x', tight=True)
plt.autoscale(enable=True, axis='y', tight=True)

print('\nProgram paused. Press enter to continue.\n')
input("Press Enter to continue...")

## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in 
#  costFunction.m

#  Setup the data matrix appropriately, and add ones for the intercept term
n = X.shape[1]

# Add intercept term to x and X_test
temp = np.ones((m,1))
X = np.c_[temp,X] 

# Initialize fitting parameters
initial_theta = np.zeros((n + 1, 1))

# Compute and display initial cost and gradient
cost = costFunction(initial_theta, X, y)
grad = gradFunction(initial_theta, X, y)

print('Cost at initial theta (zeros): {0}'.format(cost))
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros):')
print(' {0} '.format(grad))
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([ [-24], [0.2], [0.2] ])

cost = costFunction(test_theta, X, y)
grad = gradFunction(test_theta, X, y)

print('\n Cost at test theta: {0}'.format(cost))
print('Expected cost (approx): 0.218')
print('Gradient at test theta: ')
print(' {0} '.format(grad))
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

print('\nProgram paused. Press enter to continue.\n')
input("Press Enter to continue...")

### ============= Part 3: Optimizing using fminunc  =============

temp = opt.fmin_tnc(func = costFunction, 
                    x0 = initial_theta.flatten(),fprime = gradFunction, 
                    args = (X, y.flatten()))

#the output of above function is a tuple whose first element #contains the optimized values of theta
theta_optimized = temp[0]
theta_optimized = theta_optimized.reshape((n+1,1))

cost = costFunction(theta_optimized, X, y)

print('Cost at theta found by fmin_tnc: {0}\n'.format(cost))
print('Expected cost (approx): 0.203\n')
print('theta: \n')
print(' {0} \n'.format(theta_optimized))
print('Expected theta (approx):\n')
print(' -25.161\n 0.206\n 0.201\n')

# Plot Boundary
plotDecisionBoundary(theta_optimized, X, y)
plt.legend(["Decision Boundary", "Admitted", "Not Admitted"], loc='best', \
    shadow=False, fontsize='small')
# Put some labels 

# Labels and Legend
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')

## ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and 
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of 
#  our model.
#
#  Your task is to complete the code in predict.m

#  Predict probability for a student with score 45 on exam 1 
#  and score 85 on exam 2 

prob = sigmoid(np.array([[1, 45, 85]]) @ theta_optimized)
print('For a student with scores 45 and 85, we predict an admission ' \
         'probability of {0}'.format(prob))

print('Expected value: 0.775 +/- 0.002\n')

# Compute accuracy on our training set
p = predict(theta_optimized, X)

print('Train Accuracy: {0}\n'.format(np.mean((p == y).astype(float)) * 100))
print('Expected accuracy (approx): 89.0\n')

plt.show()