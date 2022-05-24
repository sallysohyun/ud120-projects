#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

t_train = time()
clf = SVC(kernel="rbf", C=10000, gamma="auto")
# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]
clf.fit(features_train, labels_train)
print("training time:")
print(round(time()-t_train, 3), "s")
# Original training time with kernel="linear" : 81.363 seconds
# Training time with kernel="linear" and 1% of training data: 0.05 seconds
# Training time with kernel="rbf", gamma="auto" and 1% of training data 
# C = 1 : 0.055 seconds
# C = 10 : 0.056 seconds
# C = 100 : 0.054 seconds
# C = 1000 : 0.050 seconds
# C = 10000 : 0.052 seconds
# Training time with kernel="rbf", C=10000: 61.26 seconds

t_predict = time()
pred = clf.predict(features_test)
score = accuracy_score(pred, labels_test)
print("Accuracy score: ", score)
print("Predicting Time:", round(time()-t_predict, 3), "s")

# print("10th: ", pred[10], ", 26th: ", pred[26], ", 50th: ", pred[50])
count = 0
for i in pred:
    if i == 1:
        count += 1

print("The predicted number of emails by Chris is: ", count)
# 877 with full training data, 1018 with 1% training data  

# Original prediction time: 17.147 seconds, Accuracy score: 0.9835
# Prediction time with kernel="linear" and 1% of training data: 0.381 seconds, Accuracy score: 0.8845
# Prediction time with kernel="rbf", gamma="auto" and 1% of training data
# C = 1 : 0.867 seconds, Accuracy: 0.616
# C = 10 : 0.929 seconds, Accuracy: 0.616
# C = 100 : 0.867 seconds, Accuracy: 0.616
# C = 1000 : 0.889 seconds, Accuracy: 0.820
# C = 10000 : 0.748 seconds, Accuracy: 0.892
# Prediction time with kernel="rbf", C=10000: 10.601 seconds, Accuracy score: 0.9903



#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
