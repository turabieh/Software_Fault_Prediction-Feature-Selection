# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 12:46:20 2019

@author: Ibrahim Aljarah, and Ruba Abu Khurma 
"""

class solution:
    def __init__(self):
        self.best = 0
        self.features = 0
        self.bestIndividual=[]
        self.convergence1 = []
        self.convergence2 = []
        self.optimizer=""
        self.objfname=""
        self.startTime=0
        self.endTime=0
        self.executionTime=0
        self.lb=0
        self.ub=0
        self.dim=0
        self.popnum=0
        self.maxiers=0
        self.trainAcc=None
        self.testAcc=None
        self.trainTP=None
        self.trainFN=None
        self.trainFP=None
        self.trainTN=None
        self.testTP=None
        self.testFN=None
        self.testFP=None
        self.testTN=None
        self.test_P=None
        self.test_N=None
        self.testTPR=None
        self.testTNR=None
        self.testAUC=None
        self.train_P=None
        self.train_N=None
        self.trainTPR=None
        self.trainTNR=None
        self.trainAUC=None
        



        
