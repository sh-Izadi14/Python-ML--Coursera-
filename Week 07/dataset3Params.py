import numpy as np
from sklearn import svm

from gaussianKernel import gaussian

def dataset3Params(X, y, Xval, yval):

#DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
#where you select the optimal (C, sigma) learning parameters to use for SVM
#with RBF kernel
#   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
#   sigma. You should complete this function to return the optimal C and 
#   sigma based on a cross-validation set.
#

# You need to return the following variables correctly.
      C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
      sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
# ====================== YOUR CODE HERE ======================
# Instructions: Fill in this function to return the optimal C and sigma
#               learning parameters found using the cross validation set.
#               You can use svmPredict to predict the labels on the cross
#               validation set. For example, 
#                   predictions = svmPredict(model, Xval);
#               will return the predictions on the cross validation set.
#
#  Note: You can compute the prediction error using 
#        mean(double(predictions ~= yval))
#

      cv_error = np.zeros((len(C),len(sigma)))

      for i in range(len(C)):
            for j in range(len(sigma)):

                  model = svm.SVC(kernel=gaussian(sigma=sigma[j]), C=C[i])
                  model.fit(X,y.flatten())

                  predictions = model.predict(Xval)
                  cv_error[i,j] = np.mean(predictions != yval.flatten())
  
      c,s = np.where(cv_error == np.min(cv_error))

      C = C[c[0]]
      sigma = sigma[s[0]]

      return C, sigma
# =========================================================================