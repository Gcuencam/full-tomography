# -*- coding: utf-8 -*-
import time
import numpy as np
from RBM import RBM
from configparser import ConfigParser

visible_layers = 3
hidden_layers = 3
learning_rate = 0.1
epochs = 100
batch_size = 10
k = 1  # Number of iterations of Gibbs Sampling.
dataset_filename = 'training.npy'
network_file_path = 'setup.ini'

dataset = np.load(dataset_filename)

# Creates the network.
rbm = RBM(visible_layers, hidden_layers)

# Trains the network.
t = time.time()
nV, nH, w, b, c = rbm.CD_K(dataset, epochs, batch_size, k, learning_rate)
print('Elapsed time:', time.time() - t)

# Saves the network.
config = ConfigParser()
config.add_section('units')
config.set('units', 'nV', str(nV))
config.set('units', 'nH', str(nH))
config.add_section('weights')
for i, v in enumerate(w):
    config.set('weights', str(i), ' '.join(map(str, v)))
config.add_section('bias')
config.set('bias', 'b', ' '.join(map(str, b)))
config.set('bias', 'c', ' '.join(map(str, c)))
with open(network_file_path, 'w') as f:
    config.write(f)

