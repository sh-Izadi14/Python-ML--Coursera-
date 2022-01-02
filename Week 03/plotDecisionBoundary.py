import numpy as np
import matplotlib.pyplot as plt
from plotData import plotData
from mapFeature import mapFeaturePlot

def plotDecisionBoundary(theta, X, y, **kwargs):
#PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
#the decision boundary defined by theta
#   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
#   positive examples and o for the negative examples. X is assumed to be 
#   a either 
#   1) Mx3 matrix, where the first column is an all-ones column for the 
#      intercept.
#   2) MxN, N>3 matrix, where the first column is all-ones
    m, n = np.shape(X)
# Plot Data
    plotData(X[:,[1,2]], y)

    if n <= 3:
    # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([np.min(X[:,1])-2,  np.max(X[:,1])+2])
    # Calculate the decision boundary line
        plot_y = (-1/theta[2,0])*(theta[1,0]*plot_x + theta[0,0])

    # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y)
    
    # Legend, specific for the exercise

    else:
    # Here is the grid range
        u_vals = np.linspace(-1,1.5,50)
        v_vals= np.linspace(-1,1.5,50)
        z=np.zeros((len(u_vals),len(v_vals)))
        for i in range(len(u_vals)):
             for j in range(len(v_vals)):                
                 z[i,j] = mapFeaturePlot(u_vals[i],v_vals[j],kwargs.get('degree')) @ theta
        plt.contour(u_vals,v_vals,z.T,0)
        
