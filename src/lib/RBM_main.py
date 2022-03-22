# -*- coding: utf-8 -*-
import numpy as np
from RBM import RBM
import time

visible_layers = 3
hidden_layers = 3
learning_rate = 0.1
epochs = 100
batch_size = 10
k = 1 #Number of iterations of Gibbs Sampling.
dataset_filename = 'training.npy'

dataset = np.load(dataset_filename)



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