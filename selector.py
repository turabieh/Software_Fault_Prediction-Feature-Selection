# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 12:46:20 2019

@author: Ibrahim Aljarah, and Ruba Abu Khurma 
"""

import PSO as pso
import MVO as mvo
import GWO as gwo
import MFO as mfo
import BAT as bat
import WOA as woa
import FFA as ffa
import CS as cs
import HHO as hho
import JAYA as jaya
import GA as ga
import HWOA as hwoa
import WOA_GWO as woa_gwo
import csv
import numpy
import time
from sklearn.model_selection import train_test_split
import pandas as pd
import fitnessFUNs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from collections import Counter



def selector(algo,func_details,popSize,Iter,completeData):
    function_name=func_details[0]
    lb=func_details[1]
    ub=func_details[2]
   
    
    #DatasetSplitRatio=0.34   #Training 66%, Testing 34%
    DatasetSplitRatio = 2/3 
    TestSize = 1 - DatasetSplitRatio
    
    DataFile="datasets/"+completeData
      
    data_set=numpy.loadtxt(open(DataFile,"rb"),delimiter=",",skiprows=0)
    numRowsData=numpy.shape(data_set)[0]    # number of instances in the  dataset
    numFeaturesData=numpy.shape(data_set)[1]-1 #number of features in the  dataset

    dataInput=data_set[0:numRowsData,0:-1]
    dataTarget=data_set[0:numRowsData,-1]  
    
        #---------------- Smote
    #counter = Counter(trainOutput)
    #print(counter)
    #sampling_strategy=0.8
    oversample = BorderlineSMOTE(sampling_strategy=0.95, random_state=42)
    dataInput, dataTarget = oversample.fit_resample(dataInput, dataTarget)
    #counter = Counter(trainOutput)
    #print(counter)
    #----------------------
    
    trainInput, testInput, trainOutput, testOutput = train_test_split(dataInput, dataTarget, test_size=TestSize, random_state=42) 
    
    #---------------- Smote
    #counter = Counter(trainOutput)
    #print(counter)
    #oversample = SMOTE(sampling_strategy=0.9, random_state=1)
    #trainInput, trainOutput = oversample.fit_resample(trainInput, trainOutput)
    #counter = Counter(trainOutput)
    #print(counter)
    #----------------------
    
#
   
#    numRowsTrain=numpy.shape(trainInput)[0]    # number of instances in the train dataset
#    numFeaturesTrain=numpy.shape(trainInput)[1]-1 #number of features in the train dataset
#
#    numRowsTest=numpy.shape(testInput)[0]    # number of instances in the test dataset
#    numFeaturesTest=numpy.shape(testInput)[1]-1 #number of features in the test dataset
# 

    dim=numFeaturesData
    
    if(algo==0):
        x=pso.PSO(getattr(fitnessFUNs, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,testInput, testOutput)
    if(algo==1):
        x=mvo.MVO(getattr(fitnessFUNs, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,testInput, testOutput)
    if(algo==2):
        x=gwo.GWO(getattr(fitnessFUNs, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,testInput, testOutput)
    if(algo==3):
        x=mfo.MFO(getattr(fitnessFUNs, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,testInput, testOutput)
    if(algo==4):
        x=woa.WOA(getattr(fitnessFUNs, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,testInput, testOutput)
    if(algo==5):
        x=ffa.FFA(getattr(fitnessFUNs, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,testInput, testOutput)
    if(algo==6):
        x=bat.BAT(getattr(fitnessFUNs, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,testInput, testOutput)
    if(algo==7):
        x=woa_gwo.WOA_GWO(getattr(fitnessFUNs, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,testInput, testOutput)
    if(algo==8):
        x=cs.CS(getattr(fitnessFUNs, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,testInput, testOutput)
    if(algo==9):
        x=hho.HHO(getattr(fitnessFUNs, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,testInput, testOutput)
    if(algo==10):
        x=jaya.JAYA(getattr(fitnessFUNs, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,testInput, testOutput)
    if(algo==11):
        x=hwoa.HWOA(getattr(fitnessFUNs, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,testInput, testOutput)
    if(algo==12):
        x=ga.GA(getattr(fitnessFUNs, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,testInput, testOutput)

    # Evaluate MLP classification model based on the training set
#    trainClassification_results=evalNet.evaluateNetClassifier(x,trainInput,trainOutput,net)
 #   x.trainAcc=trainClassification_results[0]
  #  x.trainTP=trainClassification_results[1]
   # x.trainFN=trainClassification_results[2]
    #x.trainFP=trainClassification_results[3]
    #x.trainTN=trainClassification_results[4]
   
    # Evaluate MLP classification model based on the testing set   
    #testClassification_results=evalNet.evaluateNetClassifier(x,testInput,testOutput,net)
            
    reducedfeatures=[]
    for index in range(0,dim):
        if (x.bestIndividual[index]==1):
            reducedfeatures.append(index)
    reduced_data_train_global=trainInput[:,reducedfeatures]
    reduced_data_test_global=testInput[:,reducedfeatures]
    
    #----------------------------------------------
    #knn = KNeighborsClassifier(n_neighbors=5)
    #knn.fit(reduced_data_train_global,trainOutput)
    
    #clf = RandomForestClassifier(n_estimators=100) #200 means how many trees in the forest
    clf = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=7, min_samples_split=2, min_samples_leaf=1, bootstrap=True)
    clf.fit(reduced_data_train_global, trainOutput)  # don't forget X_train features and y_train is the consumer-segment
    
    # Compute the accuracy of the prediction
         
    target_pred_train = clf.predict(reduced_data_train_global)
    #----------------------------------------------
    #----------------------------
    ConfMatrix_train=confusion_matrix(trainOutput, target_pred_train)
    ConfMatrix1D_train=ConfMatrix_train.flatten()
    x.trainTP=ConfMatrix1D_train[0]
    x.trainFN=ConfMatrix1D_train[1]
    x.trainFP=ConfMatrix1D_train[2]
    x.trainTN=ConfMatrix1D_train[3]
    x.train_P=ConfMatrix1D_train[0]+ConfMatrix1D_train[1]
    x.train_N=ConfMatrix1D_train[2]+ConfMatrix1D_train[3]
    x.trainTPR=ConfMatrix1D_train[0]/x.train_P
    x.trainTNR=ConfMatrix1D_train[3]/x.train_N
    x.trainAUC= 0.5 * (x.trainTPR+x.trainTNR)
    #----------------------------
    acc_train = float(accuracy_score(trainOutput, target_pred_train))
    x.trainAcc=acc_train
    
    target_pred_test = clf.predict(reduced_data_test_global)
    #----------------------------
    ConfMatrix_test=confusion_matrix(testOutput, target_pred_test)
    ConfMatrix1D_test= ConfMatrix_test.flatten()
    x.testTP=ConfMatrix1D_test[0]
    x.testFN=ConfMatrix1D_test[1]
    x.testFP=ConfMatrix1D_test[2]
    x.testTN=ConfMatrix1D_test[3] 
    x.test_P=ConfMatrix1D_test[0]+ConfMatrix1D_test[1]
    x.test_N=ConfMatrix1D_test[2]+ConfMatrix1D_test[3]
    x.testTPR=ConfMatrix1D_test[0]/x.test_P
    x.testTNR=ConfMatrix1D_test[3]/x.test_N
    x.testAUC= 0.5 * (x.testTPR+x.testTNR)
    #----------------------------
    acc_test = float(accuracy_score(testOutput, target_pred_test))
    x.testAcc=acc_test
    
        #print('Test set accuracy: %.2f %%' % (acc * 100))

    #x.testTP=testClassification_results[1]
    #x.testFN=testClassification_results[2]
    #x.testFP=testClassification_results[3]
    #x.testTN=testClassification_results[4] 
    
    
    return x
    
#####################################################################    
