# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 12:46:20 2019

@author: Ibrahim Aljarah, and Ruba Abu Khurma 
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Binarizer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
          
#____________________________________________________________________________________       
def FN1(I,trainInput,trainOutput,dim,testInput, testOutput):            
    #data_train_internal, data_test_internal, target_train_internal, target_test_internal = train_test_split(trainInput, trainOutput, test_size=0.34, random_state=1)
    data_train_internal = trainInput
    target_train_internal = trainOutput
    data_test_internal = testInput
    target_test_internal = testOutput
    
    reducedfeatures=[]
    for index in range(0,dim):
        if (I[index]==1):
            reducedfeatures.append(index)

    reduced_data_train_internal=data_train_internal[:,reducedfeatures]
    reduced_data_test_internal=data_test_internal[:,reducedfeatures]
    #--------------------------------------------------------------              
    #knn = KNeighborsClassifier(n_neighbors=5)
    #knn.fit(reduced_data_train_internal, target_train_internal)
    
    #clf = RandomForestClassifier(n_estimators=100) #200 means how many trees in the forest
    clf = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=7, min_samples_split=2, min_samples_leaf=1, bootstrap=True) #200 means how many trees in the forest
    clf.fit(reduced_data_train_internal, target_train_internal)  # don't forget X_train features and y_train is the consumer-segment
    target_pred_internal = clf.predict(reduced_data_test_internal)
    #--------------------------------------------------------------
    #acc_train = float(accuracy_score(target_test_internal, target_pred_internal))
     
    auc_train = float(roc_auc_score(target_test_internal,target_pred_internal))
       
    #fitness=0.99*(1-acc_train)+0.01*sum(I)/(dim)
    fitness=0.99*(1-auc_train)+0.01*sum(I)/(dim)

    return fitness
#_____________________________________________________________________       
def getFunctionDetails(a):
    
    # [name, lb, ub, dim]
    param = {  0:["FN1",0,1]

            }
    return param.get(a, "nothing")



