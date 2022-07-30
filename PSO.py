# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 12:46:20 2019

@author: Ibrahim Aljarah, and Ruba Abu Khurma 
"""

import random
import numpy
import math
from colorama import Fore, Back, Style
from solution import solution
import time
from sklearn.preprocessing import Binarizer
import transfer_functions_benchmark
import fitnessFUNs
from random import seed





def PSO(objf,lb,ub,dim,PopSize,iters,trainInput,trainOutput,testInput,testOutput):

    # PSO parameters
    
    Vmax=6
    wMax=0.9
    wMin=0.2
    c1=2
    c2=2
    lb=0
    ub=1   
    s=solution()
    
    
    ######################## Initializations
    
    vel=numpy.zeros((PopSize,dim))
    
    pBestScore=numpy.zeros(PopSize) 
    pBestScore.fill(float("inf"))
    
    pBest=numpy.zeros((PopSize,dim))
    gBest=numpy.zeros(dim)
    
    
    gBestScore=float("inf")
    
    pos=numpy.random.randint(2, size=(PopSize,dim)) #generating binary individuals
    convergence_curve1=numpy.zeros(iters)
    convergence_curve2=numpy.zeros(iters)

    print("ssssssssssssssssssss")
    ############################################
    print("PSO is optimizing  \""+objf.__name__+"\"")    
    
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    
    for l in range(0,iters):
        for i in range(0,PopSize):
            # the following statement insures that at least one feature is selected
            #(i.e the randomly generated individual has at least one value 1)
            while numpy.sum(pos[i,:])==0:   
                 pos[i,:]=numpy.random.randint(2, size=(1,dim))
                 
            #Calculate objective function for each particle
            fitness=objf(pos[i,:],trainInput,trainOutput,dim,testInput, testOutput)
            print ("fitness = "+ str(fitness))

            if(pBestScore[i]>fitness):
                pBestScore[i]=fitness
                pBest[i,:]=pos[i,:]

            if(gBestScore>fitness):
                gBestScore=fitness #best fitness on training returned from F10
                gBest=pos[i,:]

         
        #featurecount=0    # ADDED BY THAER
        #featurecount=sum(gBest)            
        #for f in range(0,dim):
        #    if gBest[f]==1:
        #        featurecount=featurecount+1
        #print("featurecount = " + str(featurecount))
        convergence_curve2[l]=sum(gBest)# store the best number of features
        convergence_curve1[l]=gBestScore#store the best fitness on testing returened from F11

        #if (l%1==0):
        print(['iteration'+ str(l+1)+'best fitness  is:'+ str(gBestScore)+', the best number of features: '+str(sum(gBest))]);     
        
        #Update the W of PSO
        w=wMax-l*((wMax-wMin)/iters);
        
        for i in range(0,PopSize):
            for j in range (0,dim):
                r1=random.random()
                r2=random.random()
                vel[i,j]=w*vel[i,j]+c1*r1*(pBest[i,j]-pos[i,j])+c2*r2*(gBest[j]-pos[i,j])
                
                if(vel[i,j]>Vmax):
                    vel[i,j]=Vmax
                
                if(vel[i,j]<-Vmax):
                    vel[i,j]=-Vmax
                            
                pos[i,j]=(pos[i,j]+vel[i,j])#update statement
                
                ss= transfer_functions_benchmark.s1(pos[i,j])#transfer function
                #print(transfer_functions_benchmark.s1(pos[i,j]))
                #time.sleep(2)   
                
                if (random.random()<ss): 
                    pos[i,j]=1;
                else:
                    pos[i,j]=0;
            
               # print(("jjjjj"+str(pos[i,j])))

            
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.bestIndividual=gBest
    s.convergence1=convergence_curve1
    s.convergence2=convergence_curve2
    s.best = gBestScore
    s.features = sum(gBest)
    s.optimizer="PSO"
    s.objfname=objf.__name__

    return s
         
    
