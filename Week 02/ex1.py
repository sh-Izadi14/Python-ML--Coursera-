## Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     warmUpExercise.py
#     plotData.py
#     GradientDescent.py
#     ComputeCost.py
#     GradientDescentMulti.py
#     ComputeCostMulti.py
#     featureNormalize.py
#     normalEqn.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s
#

## Initialization
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from warmup import warmUpExercise
from plotData import plotData
from computeCost import computeCost
from gradientDescent import batchGD

## ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.py
print('Running warmUpExercise ...')
print('5x5 Identity Matrix: ')
warmUpExercise()

input("Press Enter to continue...")

## ======================= Part 2: Plotting =======================
print('Plotting Data ...')
data1 = np.loadtxt('ex1data1.txt', delimiter = ",")
x = data1[:,0]
y = data1[:,1]
m = len(y)

# Plot Data
# Note: You have to complete the code in plotData.py
plotData(x, y)
plt.show()

input("Press Enter to continue...")

## =================== Part 3: Cost and Gradient descent ===================
temp = np.ones((m,1))
X = np.c_[temp,x]  # Add a column of ones to x
y = y.reshape((m,1))

theta = np.zeros((X.shape[1],1)) # initialize fitting parameters

# compute and display initial cost
J = computeCost(X, y, theta)
print('\nWith theta = [0 ; 0]\nCost computed ={0}'.format(J[0]))
print('Expected cost value (approx) 32.07')

# further testing of the cost function
J = computeCost(X, y, np.array([[-1],[2]]))
print('\nWith theta = [-1 ; 2]\nCost computed ={0}'.format(J[0]))
print('Expected cost value (approx) 54.24')

# Some gradient descent settings
iterations = 1500
alpha = 0.01

# run gradient descent
theta, J_history, iter = batchGD(X, y, theta, alpha, iterations)

# print theta to screen
print("\nTheta found by gradient descent:")
print("{0}\n{1}".format(theta[0],theta[1]) )
print('Expected theta values (approx):')
print('-3.6303\n  1.1664')

# Plot the linear fit
pred = X @ theta

plotData(x, y)
plt.plot(X[:,1], pred, '-', label = 'Linear regression')
plt.legend()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.matmul(np.array([1, 3.5]), theta)
print("For population = 35000, we predict a profit of {0}".format(predict1*10000))

predict2 = np.matmul(np.array([1, 7]), theta)
print("For population = 70000, we predict a profit of {0}".format(predict2*10000))

input("Press Enter to continue...")

# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print("'Visualizing J(theta_0, theta_1) ...")

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100, endpoint = True)
theta1_vals = np.linspace(-1, 4, 100, endpoint = True)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Fill out J_vals
for i in range(len(theta0_vals)):
	for j in range(len(theta1_vals)):
		t = np.array([ [theta0_vals[i]], [theta1_vals[j]] ])
		J_vals[i,j] = computeCost(X, y, t)

# Surface plot
fig = plt.figure(2)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(theta0_vals, theta1_vals, J_vals.T, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')

# Contour plot
plt.figure(3)
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100

plt.contour(theta0_vals, theta1_vals, J_vals.T, np.logspace(-2, 3, 20))
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')

plt.plot(theta[0], theta[1], 'rx')

plt.show()