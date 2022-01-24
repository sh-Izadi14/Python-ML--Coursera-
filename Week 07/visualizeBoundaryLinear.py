import numpy as np
import matplotlib.pyplot as plt
from plotData import plotData

def visualizeBoundaryLinear(X, y, model):
#VISUALIZEBOUNDARYLINEAR plots a linear decision boundary learned by the
#SVM
#   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a linear decision boundary 
#   learned by the SVM and overlays the data on it

      X_1,X_2 = np.meshgrid(np.linspace(X[:,0].min(),X[:,1].max(),num=1001),np.linspace(X[:,1].min(),X[:,1].max(),num=1001))
      

      plotData(X,y)

      plt.contour(X_1,X_2,model.predict(np.array([X_1.ravel(),X_2.ravel()]).T).reshape(X_1.shape),1,colors="b")

      plt.xlim(0,4.5)
      plt.ylim(1.5,5)
      