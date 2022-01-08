import numpy as np
import scipy.optimize as opt

from linearRegCostFunction import linearRegCostFunction, linearRegGradientFunction


def trainLinearReg(X, y, Lambda):
#TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
#regularization parameter lambda
#   [theta] = TRAINLINEARREG (X, y, lambda) trains linear regression using
#   the dataset (X, y) and regularization parameter lambda. Returns the
#   trained parameters theta.
#

# Initialize Theta
      initial_theta = np.zeros((X.shape[1], 1))
      res = opt.minimize(fun = linearRegCostFunction, x0 = initial_theta.flatten(), \
                        args = (X, y.flatten(), Lambda), method='TNC',\
                        jac = linearRegGradientFunction,\
                        options={'disp':True, 'maxiter': 100})
      theta = res.x
      return theta
