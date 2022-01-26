import matplotlib.pyplot as plt
import numpy as np

def drawLine(p1, p2, **kwargs):
#DRAWLINE Draws a line from point p1 to point p2
#   DRAWLINE(p1, p2) Draws a line from point p1 to point p2 and holds the
#   current figure
      
      point1 = np.array([p1[0], p2[0]])
      point2 = np.array([p1[1], p2[1]])

      if not kwargs.get('color'):
            color = 'k'
      else:
            color = kwargs.get('color')

      plt.plot(point1, point2,linewidth = kwargs.get('linewidth'), \
            linestyle = kwargs.get('linestyle'), c = color)
      