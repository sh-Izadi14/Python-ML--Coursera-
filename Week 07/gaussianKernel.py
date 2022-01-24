import numpy as np

def gaussianKernel(x1, x2, sigma):

#RBFKERNEL returns a radial basis function kernel between x1 and x2
#   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
#   and returns the value in sim

# Ensure that x1 and x2 are column vectors
      x1 = x1.reshape((x1.size,1))
      x2 = x2.reshape((x2.size,1))

# You need to return the following variables correctly.
      sim = 0

# ====================== YOUR CODE HERE ======================
# Instructions: Fill in this function to return the similarity between x1
#               and x2 computed using a Gaussian kernel with bandwidth
#               sigma
#
#

      numerator = (x1 - x2).T @ (x1 - x2)

      sim  = np.exp(-numerator/(2*sigma**2))

      return sim

# =============================================================

def gaussian(sigma):

    def k_gaussian(_x1, _x2):
        diff = _x1[:, np.newaxis] - _x2
        normsq = np.square(np.linalg.norm(diff, axis = 2))
        return np.exp(- normsq / (2 * np.square(sigma)))

    return k_gaussian