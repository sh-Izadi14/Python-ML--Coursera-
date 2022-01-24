## Machine Learning Online Class
#  Exercise 6 | Support Vector Machines
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     gaussianKernel.m
#     dataset3Params.m
#     processEmail.m
#     emailFeatures.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Initialization
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn import svm

from plotData import plotData
from visualizeBoundaryLinear import visualizeBoundaryLinear
from gaussianKernel import gaussianKernel, gaussian
from visualizeBoundary import visualizeBoundary
from dataset3Params import dataset3Params

## =============== Part 1: Loading and Visualizing Data ================
#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.
#

print('Loading and Visualizing Data ...\n')

# Load from ex6data1: 
# You will have X, y in your environment
mat = scipy.io.loadmat('ex6data1.mat')
X = mat['X']
y = mat['y']

# Plot training data
plotData(X, y)
plt.show()

print('Program paused.')
input('Press enter to continue...')

## ==================== Part 2: Training Linear SVM ====================
#  The following code will train a linear SVM on the dataset and plot the
#  decision boundary learned.
#

print('\nTraining Linear SVM ...\n')

# You should try to change the C value below and see how the decision
# boundary varies (e.g., try C = 1000)
C = 1
model = svm.SVC(kernel="linear", C = C)
model.fit(X,y.flatten())
visualizeBoundaryLinear(X, y, model)
plt.show()

print('Program paused.')
input('Press enter to continue...')


## =============== Part 3: Implementing Gaussian Kernel ===============
#  You will now implement the Gaussian kernel to use
#  with the SVM. You should complete the code in gaussianKernel.m
#
print('\nEvaluating the Gaussian Kernel ...\n')

x1 = np.array([[1, 2, 1]])
x2 = np.array([[0, 4, -1]])
sigma = 2
sim = gaussianKernel(x1, x2, sigma)

print('Gaussian Kernel between x1 = [1 2 1], x2 = [0 4 -1], sigma = {} :'\
         '\n\t#{}\n(for sigma = 2, this value should be about 0.324652)\n'.format(sigma, sim))

print('Program paused.')
input('Press enter to continue...')

## =============== Part 4: Visualizing Dataset 2 ================
#  The following code will load the next dataset into your environment and 
#  plot the data. 
#

print('Loading and Visualizing Data ...\n')

# Load from ex6data2: 
# You will have X, y in your environment
mat2 = scipy.io.loadmat('ex6data2.mat')
X2 = mat2['X']
y2 = mat2['y']

# Plot training data
plotData(X2, y2)
plt.show()

print('Program paused.')
input('Press enter to continue...')

## ========== Part 5: Training SVM with Gaussian Kernel (Dataset 2) ==========
#  After you have implemented the kernel, we can now use it to train the 
#  SVM classifier.
# 
print('\nTraining SVM with Gaussian Kernel (this may take 1 to 2 minutes) ...\n')

# SVM Parameters
C = 1 
sigma = 0.1

# We set the tolerance and max_passes lower here so that the code will run
# faster. However, in practice, you will want to run the training to
# convergence.
model2 = svm.SVC(kernel=gaussian(sigma=sigma), C=C)
model2.fit(X2,y2.flatten())
visualizeBoundary(X2, y2, model2)
plt.show()

print('Program paused.')
input('Press enter to continue...')

## =============== Part 6: Visualizing Dataset 3 ================
#  The following code will load the next dataset into your environment and 
#  plot the data. 
#

print('Loading and Visualizing Data ...\n')

# Load from ex6data3: 
# You will have X, y in your environment
mat3 = scipy.io.loadmat('ex6data3.mat')
X3 = mat3['X']
y3 = mat3['y']
Xval = mat3['Xval']
yval = mat3['yval']
# Plot training data
plotData(X3, y3)
plt.show()

print('Program paused.')
input('Press enter to continue...')

## ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

#  This is a different dataset that you can use to experiment with. Try
#  different values of C and sigma here.
# 

# Try different SVM Parameters here
C, sigma = dataset3Params(X3, y3, Xval, yval)
print(f'\nbest C: {C}\n',f'best sigma: {sigma}')

# Train the SVM
model3= svm.SVC(kernel=gaussian(sigma=sigma), C=C)
model3.fit(X3,y3.flatten())

visualizeBoundary(X3, y3, model3)
plt.show()

