#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
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


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

t_train = time()
clf = DecisionTreeClassifier(min_samples_split=40)

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]
clf.fit(features_train, labels_train)
print("training time:")
print(round(time()-t_train, 3), "s")

t_predict = time()
pred = clf.predict(features_test)
score = accuracy_score(pred, labels_test)
print("Accuracy score: ", score)
print("Predicting Time:", round(time()-t_predict, 3), "s")


print("Number of features: ", len(features_train[0]))
# 10% feature: 3785
# 1% feature: 379 


#########################################################


