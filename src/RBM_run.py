# -*- coding: utf-8 -*-
import numpy as np
from RBM import RBM
from configparser import ConfigParser


network_file_path = 'setup5h.ini'


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


# Overlap calculation.
basisW = np.identity(nV) #Basis of the W state.
basisGHZ = np.array([[0,0,0],[1,1,1]])
n = 1000  # Number of iterations of the overlap's sampling.
k = 5  # Number of iterations of Gibbs Sampling.

o = rbm.Overlap(basisW, n, k)
print('O^2 = ' + str(o))