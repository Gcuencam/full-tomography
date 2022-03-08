# -*- coding: utf-8 -*-
import numpy as np
from RBM import RBM
import time

visible_layers = 3
hidden_layers = 2
learning_rate = 0.2
epochs = 1000
batch_size = 10
k = 1 #Number of iterations of Gibbs Sampling.
m = 1728
n = 3
dataset_filename = 'test.txt'

dataset = np.loadtxt(dataset_filename)



#Creates the network.
rbm = RBM(visible_layers,hidden_layers)

#Trains the network.
t=time.time()
rbm.CD_K(dataset,epochs,batch_size,k,learning_rate)
print('Elapsed time:',time.time()-t)

#Runs the network. This reconstructs the input sample.
newData = [0,0,0]
reconstruction = rbm.run(newData,k)
print('Reconstruction of the new data:',reconstruction)

