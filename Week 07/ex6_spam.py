## Machine Learning Online Class
#  Exercise 6 | Spam Classification with SVMs
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
import scipy.io
from sklearn import svm

from processEmail import processEmail
from emailFeatures import emailFeatures
from getVocabList import getVocabList
## ==================== Part 1: Email Preprocessing ====================
#  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
#  to convert each email into a vector of features. In this part, you will
#  implement the preprocessing steps for each email. You should
#  complete the code in processEmail.m to produce a word indices vector
#  for a given email.

print('\nPreprocessing sample email (emailSample1.txt)\n')

# Extract Features

file_contents = open("emailSample1.txt","r").read()
word_indices  = processEmail(file_contents)

# Print Stats
print('\n\nWord Indices: ')
print(f'{word_indices}')
print()

print('Program paused.')
input('Press enter to continue.')

## ==================== Part 2: Feature Extraction ====================
#  Now, you will convert each email into a vector of features in R^n. 
#  You should complete the code in emailFeatures.m to produce a feature
#  vector for a given email.

filename = 'emailSample1.txt'

print(f'\nExtracting features from sample email ({filename})\n')

# Extract Features
file_contents = open(filename,"r").read()
word_indices  = processEmail(file_contents)
features      = emailFeatures(word_indices)

# Print Stats
print('\n\nLength of feature vector: {}\n'.format(len(features)))
print('Number of non-zero entries: {}\n'.format(sum(features > 0)))

print('Program paused.')
input('Press enter to continue.')

## =========== Part 3: Train Linear SVM for Spam Classification ========
#  In this section, you will train a linear classifier to determine if an
#  email is Spam or Not-Spam.

# Load the Spam Email dataset
# You will have X, y in your environment
mat= scipy.io.loadmat('spamTrain.mat')
X = mat['X']
y = mat['y']

print('\nTraining Linear SVM (Spam Classification)')
print('(this may take 1 to 2 minutes) ...')

C = 0.1
model = svm.SVC(kernel="linear", C = C)
model.fit(X,y.flatten())

p = model.predict(X)

print('Training Accuracy: {}\n'.format(np.mean((p.flatten() == y.flatten())) * 100))
input('Press enter to continue.')

## =================== Part 4: Test Spam Classification ================
#  After training the classifier, we can evaluate it on a test set. We have
#  included a test set in spamTest.mat

# Load the test dataset
# You will have Xtest, ytest in your environment
mat= scipy.io.loadmat('spamTest.mat')
Xtest = mat['Xtest']
ytest = mat['ytest']


print('\nEvaluating the trained Linear SVM on a test set ...')

p = model.predict(Xtest)

print('Test Accuracy: {}\n'.format(np.mean((p.flatten() == ytest.flatten())) * 100))
input('Press enter to continue.')

## ================= Part 5: Top Predictors of Spam ====================
#  Since the model we are training is a linear SVM, we can inspect the
#  weights learned by the model to understand better how it is determining
#  whether an email is spam or not. The following code finds the words with
#  the highest weights in the classifier. Informally, the classifier
#  'thinks' that these words are the most likely indicators of spam.
#

# Sort the weights and obtin the vocabulary list
weight = model.coef_[0]
weight_sorted = np.sort(weight)
weight_sorted = weight_sorted[::-1]

vocabList = getVocabList()

print('\nTop predictors of spam: ')
for i in range(15):
    
    idx = int(np.where(weight == weight_sorted[i])[0])
    key_list = list(vocabList.keys())
    val_list = list(vocabList.values())
    position = val_list.index(idx)

    print('{} \t\t\t ({}) \n'.format(key_list[position], weight_sorted[i]))

print('\n')
input('\nPress enter to continue.')

## =================== Part 6: Try Your Own Emails =====================
#  Now that you've trained the spam classifier, you can use it on your own
#  emails! In the starter code, we have included spamSample1.txt,
#  spamSample2.txt, emailSample1.txt and emailSample2.txt as examples. 
#  The following code reads in one of these emails and then uses your 
#  learned SVM classifier to determine whether the email is Spam or 
#  Not Spam

# Set the file to be read in (change this to spamSample2.txt,
# emailSample1.txt or emailSample2.txt to see different predictions on
# different emails types). Try your own emails as well!
filename = 'spamSample1.txt'

# Read and predict
file_contents = open(filename, "r").read()
word_indices  = processEmail(file_contents)
x             = emailFeatures(word_indices)

p = model.predict(x.T)

print(f'\nProcessed {filename}\n\nSpam Classification: {int(p)}\n')
print('(1 indicates spam, 0 indicates not spam)\n\n')
