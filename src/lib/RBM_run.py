# -*- coding: utf-8 -*-
import numpy as np
from RBM import RBM
from configparser import ConfigParser

k = 1  # Number of iterations of Gibbs Sampling.
dataset_filename = 'src/lib/training.npy'
network_file_path = 'src/lib/setup.ini'

dataset = np.load(dataset_filename)

config = ConfigParser()
config.read(network_file_path)

# Creates the network.
nV = int(config.get('units', 'nV'))
nH = int(config.get('units', 'nH'))
_w = []
for i in range(nV):
    v = np.asarray(config.get('weights', str(i)).split(), dtype=np.float64)
    _w.append(v)
b = np.asarray(config.get('bias', 'b').split(), dtype=np.float64)
c = np.asarray(config.get('bias', 'c').split(), dtype=np.float64)
w = np.asarray(_w)

rbm = RBM(nV, nH, w, b, c)

# Runs the network. This reconstructs the input sample.
newData = [0, 0, 0]
reconstruction = rbm.run(newData, k)
print('Reconstruction of the new data:', reconstruction)
