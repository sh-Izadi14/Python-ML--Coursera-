import numpy as np
import matplotlib.pyplot as plt

def plotData(X, y):
#PLOTDATA Plots the data points X and y into a new figure 
#   PLOTDATA(x,y) plots the data points with + for the positive examples
#   and o for the negative examples. X is assumed to be a Mx2 matrix.

# Create New Figure
    plt.figure()

# ====================== YOUR CODE HERE ======================
# Instructions: Plot the positive and negative examples on a
#               2D plot, using the option 'k+' for the positive
#               examples and 'ko' for the negative examples.
#
# Find Indices of Positive and Negative Examples
    pos = np.argwhere(y==1); neg = np.argwhere(y == 0)

# Plot Examples
    plt.scatter(X[pos, 0], X[pos, 1], color = 'k', marker = '+',  \
        label = 'Admitted')

    plt.scatter(X[neg, 0], X[neg, 1], color = 'y', marker = 'o', facecolor = 'yellow', \
        label = 'Not Admitted')
