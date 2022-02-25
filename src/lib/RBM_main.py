# -*- coding: utf-8 -*-
import numpy as np
from RBM import RBM

visible_layers = 3
hidden_layers = 2
learning_rate = 0.2
epochs = 500
batch_size = 2
k = 1 #Number of iterations of Gibbs Sampling.
dataset = np.array([[0,1,0],[1,0,1],[0,1,0],[1,0,1],[0,1,0],[1,0,1],[0,1,0],[1,0,1],[0,1,0],[1,0,1]])


#Creates the network.
rbm = RBM(visible_layers,hidden_layers)

#Trains the network.
rbm.CD_K(dataset,epochs,batch_size,k,learning_rate)

#Runs the network. This reconstructs the input sample.
newData = [0,0,0]
reconstruction = rbm.run(newData,k)
print(reconstruction)

