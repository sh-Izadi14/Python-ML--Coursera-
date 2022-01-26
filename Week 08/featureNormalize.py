#FEATURENORMALIZE Normalizes the features in X 
#   FEATURENORMALIZE(X) returns a normalized version of X where
#   the mean value of each feature is 0 and the standard deviation
#   is 1. This is often a good preprocessing step to do when
#   working with learning algorithms.


# ====================== YOUR CODE HERE ======================
# Instructions: First, for each feature dimension, compute the mean
#               of the feature and subtract it from the dataset,
#               storing the mean value in mu. Next, compute the 
#               standard deviation of each feature and divide
#               each feature by it's standard deviation, storing
#               the standard deviation in sigma. 
#
#               Note that X is a matrix where each column is a 
#               feature and each row is an example. You need 
#               to perform the normalization separately for 
#               each feature. 
#
# Hint: You might find the 'mean' and 'std' functions useful.
#       

import numpy as np

def featureNormalize(X):

    mu = np.mean(X, axis = 0)
    X_norm = X - mu
    
    sigma = np.std(X_norm, axis = 0)
    X_norm = X_norm/sigma

    return X_norm, mu, sigma