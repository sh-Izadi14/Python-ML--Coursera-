import numpy as np
import matplotlib.pyplot as plt
from plotData import plotData
def visualizeBoundary(X, y, model):
#VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
#   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision 
#   boundary learned by the SVM and overlays the data on it

# Plot the training data on top of the boundary
      plotData(X, y)


     # plotting the decision boundary
      X_1,X_2 = np.meshgrid(np.linspace(X[:,0].min(),X[:,0].max(),num=100),np.linspace(X[:,1].min(),X[:,1].max(),num=100))
      plt.contour(X_1,X_2,model.predict(np.array([X_1.ravel(),X_2.ravel()]).T).reshape(X_1.shape),1,colors="b")