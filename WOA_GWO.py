# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 12:46:20 2019

@author: Ibrahim Aljarah, and Ruba Abu Khurma 
"""

import random
import numpy
import math
from solution import solution
import time
import transfer_functions_benchmark
import fitnessFUNs


def uniform_crossover(A, B, P):
    for i in range (len(P)):
        if P[i] < 0.5:
            tmp=A[i]
            A[i] = B[i]
            B[i] = tmp
    return A,B

def WOA_GWO(objf,lb,ub,dim,SearchAgents_no,Max_iter,trainInput,trainOutput,testInput, testOutput):
        
    # initialize position vector and score for the leader
    Leader_pos=numpy.zeros(dim)
    Leader_score=float("inf")  #change this to -inf for maximization problems
    
    Beta_pos=numpy.zeros(dim)
    Beta_score=float("inf")
    
    Delta_pos=numpy.zeros(dim)
    Delta_score=float("inf")
    
    
    #Initialize the positions of search agents
    #Positions=numpy.random.uniform(0,1,(SearchAgents_no,dim)) *(ub-lb)+lb #generating continuous individuals
    Positions=numpy.random.randint(2, size=(SearchAgents_no,dim))#generating binary individuals
    #Initialize convergence
    convergence_curve1=numpy.zeros(Max_iter)
    convergence_curve2=numpy.zeros(Max_iter)

    ############################
    s=solution()

    print("WOA-GWO is optimizing  \""+objf.__name__+"\"")    

    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    ############################
    t=0  # Loop counter
    
    # Main loop
    while t<Max_iter:
        for i in range(0,SearchAgents_no):
            
            # the following statement insures that at least one feature is selected
            #(i.e the randomly generated individual has at least one value 1)
            while numpy.sum(Positions[i,:])==0:   
                 Positions[i,:]=numpy.random.randint(2, size=(1,dim))
            
            # Calculate objective function for each search agent
            fitness=objf(Positions[i,:],trainInput,trainOutput,dim,testInput, testOutput);
            
            # Update the leader
            if fitness<Leader_score: # Change this to > for maximization problem
                Leader_score=fitness; # Update alpha
                Leader_pos=Positions[i,:].copy() # copy current whale position into the leader position
            
            if (fitness>Leader_score and fitness<Beta_score ):
                Beta_score=fitness  # Update beta
                Beta_pos=Positions[i,:].copy()
            
            
            if (fitness>Leader_score and fitness>Beta_score and fitness<Delta_score): 
                Delta_score=fitness # Update delta
                Delta_pos=Positions[i,:].copy()
            
            
            featurecount=0
            for f in range(0,dim):
                if Leader_pos[f]==1:
                    featurecount=featurecount+1
            
            
            convergence_curve1[t]=Leader_score
            convergence_curve2[t]=featurecount
            if (t%1==0):
                print(['At iteration '+ str(t)+ ' the best fitness on trainig is: '+ str(Leader_score)+'the best number of features: '+str(featurecount)]);
        
                
        a=2-t*((2)/Max_iter); # a decreases linearly fron 2 to 0 in Eq. (2.3)
        
        # a2 linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
        a2=-1+t*((-1)/Max_iter);
        
        # Update the Position of search agents 
        for i in range(0,SearchAgents_no):
            r1=random.random() # r1 is a random number in [0,1]
            r2=random.random() # r2 is a random number in [0,1]
            
            A1=2*a*r1-a  # Eq. (2.3) in the paper
            C1=2*r2      # Eq. (2.4) in the paper
            
            r3=random.random() # r1 is a random number in [0,1]
            r4=random.random() # r2 is a random number in [0,1]
            
            A2=2*a*r3-a  # Eq. (2.3) in the paper
            C2=2*r4      # Eq. (2.4) in the paper
            
            r5=random.random() # r1 is a random number in [0,1]
            r6=random.random() # r2 is a random number in [0,1]
            
            A3=2*a*r5-a  # Eq. (2.3) in the paper
            C3=2*r6      # Eq. (2.4) in the paper
            
            
            b=1;               #  parameters in Eq. (2.5)
            l=(a2-1)*random.random()+1   #  parameters in Eq. (2.5)
            
            p = random.random()        # p in Eq. (2.6)
            
            for j in range(0,dim):
                if abs(A1)>=1:
                        flag=1
                        #rand_leader_index = math.floor(SearchAgents_no*random.random());
                        #X_rand = Positions[rand_leader_index, :]
                        #D_X_rand=abs(C1*X_rand[j]-Positions[i,j]) 
                        #Positions[i,j]=X_rand[j]-A1*D_X_rand   #update statement
                        
                        D_Leader=abs(C1*Leader_pos[j]-Positions[i,j])
                        X1=Leader_pos[j]-A1*D_Leader
             
                        D_beta=abs(C2*Beta_pos[j]-Positions[i,j]); # Equation (3.5)-part 2
                        X2=Beta_pos[j]-A2*D_beta; # Equation (3.6)-part 2
                    
                        
                        D_delta=abs(C3*Delta_pos[j]-Positions[i,j]); # Equation (3.5)-part 3
                        X3=Delta_pos[j]-A3*D_delta; # Equation (3.5)-part 3 
                    
                        Positions[i,j]=(X1+X2+X3)/3

                    
                elif abs(A1)<1:
                        flag=0
                        if(p<0.5):
                            D_Leader=abs(C1*Leader_pos[j]-Positions[i,j])
                            Positions[i,j]=Leader_pos[j]-A1*D_Leader    #update statement 
   
                        elif p>=0.5:
                            distance2Leader=abs(Leader_pos[j]-Positions[i,j])
                            # Eq. (2.5)
                            Positions[i,j]=distance2Leader*math.exp(b*l)*math.cos(l*2*math.pi)+Leader_pos[j]
                                    
            

                
                ss= transfer_functions_benchmark.s1(Positions[i,j])
                    
                if (random.random()<ss): 
                        Positions[i,j]=1;
                else:
                        Positions[i,j]=0;
        #print(Positions[i,:])
        t=t+1
    
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.bestIndividual=Leader_pos
    s.convergence1=convergence_curve1
    s.convergence2=convergence_curve2
    s.best = Leader_score
    s.features = featurecount
    s.optimizer="WOA_GWO"
    s.objfname=objf.__name__
    
    return s


