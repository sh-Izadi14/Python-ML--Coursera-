import scipy.optimize as opt
import numpy as np
from lrCostFunction import lrCostFunction, lrGradFunction

def  oneVsAll(X, y, num_labels, Lambda):

      m, n = X.shape
      all_theta = np.zeros((num_labels, n+1))
      
      temp = np.ones((m,1))
      X = np.c_[temp, X]
      
      initial_theta = np.zeros((n+1, 1))

      for c in range(num_labels):
            y1 = (y == c)
            res = opt.fmin_tnc(func = lrCostFunction, x0 = initial_theta.flatten(), \
                   args = (X, y1.flatten(), Lambda),  fprime = lrGradFunction, messages = 0)

            all_theta[c,:] = res[0]

      return all_theta