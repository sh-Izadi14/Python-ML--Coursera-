from cProfile import label
from tokenize import PlainToken
import numpy as np
import matplotlib.pyplot as plt

def plotData(X, y):
#PLOTDATA Plots the data points X and y into a new figure 
#   PLOTDATA(x,y) plots the data points with + for the positive examples
#   and o for the negative examples. X is assumed to be a Mx2 matrix.
#
# Note: This was slightly modified such that it expects y = 1 or y = 0

# Find Indices of Positive and Negative Examples
      m = X.shape[0]
      pos = (y==1).reshape(m,1)
      neg = (y==0).reshape(m,1)

# Plot Examples
      
      plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+",s=50)
      plt.scatter(X[neg[:,0],0],X[neg[:,0],1],c="y",marker="o",s=50)
