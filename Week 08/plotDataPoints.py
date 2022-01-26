import matplotlib.pyplot as plt
import numpy as np
from generateColors import generateColors

def plotDataPoints(X, idx, K):
#PLOTDATAPOINTS plots data points in X, coloring them so that those with the same
#index assignments in idx have the same color
#   PLOTDATAPOINTS(X, idx, K) plots data points in X, coloring them so that those 
#   with the same index assignments in idx have the same color
      
      plt.scatter(X[:,0],X[:,1], facecolors='none',\
             edgecolors=generateColors(K, idx), s=15)      