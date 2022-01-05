import numpy as np
from sigmoid import sigmoid

def predictOneVsAll(all_theta, X):
      
      temp = np.ones((X.shape[0],1))
      X = np.c_[temp, X]
      
      pred = sigmoid(X @ all_theta.T)
      p = np.argmax(pred, axis=1)
      return p
