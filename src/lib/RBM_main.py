# -*- coding: utf-8 -*-
import numpy as np
from RBM import RBM

visible_layers = 3
hidden_layers = 6
learning_rate = 0.2
epochs = 500
k = 1 #Number of iterations of Gibbs Sampling.
dataset = np.array([[0,1,0],[1,0,1],[0,1,0],[1,0,1],[0,1,0],[1,0,1]])



rbm = RBM(visible_layers,hidden_layers,learning_rate)

#Trains the network.
rbm.CD_K(dataset,epochs,k)

#Runs the network. This reconstructs the input sample.
newData = [0,0,0]
reconstruction = rbm.run(newData,k)
print(reconstruction)

