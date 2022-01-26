import matplotlib.pyplot as plt
import numpy as np

from drawLine import drawLine
from plotDataPoints import plotDataPoints

def plotProgresskMeans(X, centroids, previous, idx, K, i):
# PLOTPROGRESSKMEANS is a helper function that displays the progress of 
# k-Means as it is running. It is intended for use only with 2D data.
#  PLOTPROGRESSKMEANS(X, centroids, previous, idx, K, i) plots the data
#  points with colors assigned to each centroid. With the previous
#  centroids, it also plots a line between the previous locations and
#  current locations of the centroids.
#

      #Plot the examples
      plt.ion()
      plotDataPoints(X, idx, K)
      
      #Plot the centroids as black x's
      plt.scatter(centroids[:,0], centroids[:,1], color = 'k',\
             marker = 'x', linewidths=2, s=50)

      #Plot the history of the centroids with lines
      for j in range(centroids.shape[0]):
            drawLine(centroids[j, :], previous[j, :])

      #Title
      plt.title(f'Iteration number {i+1}')
      
      plt.show()
