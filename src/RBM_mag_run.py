# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import numpy as np
import math
import time
from RBM_mag import RBM_mag
from configparser import ConfigParser


# Creates the network of magnitudes.
network_file_path = 'rbm_mag.ini'
config = ConfigParser()
config.read(network_file_path)
nV = int(config.get('units', 'nV'))
nH = int(config.get('units', 'nH'))
_w = []
for i in range(nV):
    v = np.asarray(config.get('weights', str(i)).split(), dtype=np.float64)
    _w.append(v)
b = np.asarray(config.get('bias', 'b').split(), dtype=np.float64)
c = np.asarray(config.get('bias', 'c').split(), dtype=np.float64)
w = np.asarray(_w)

rbm_mag = RBM_mag(nV, nH, w, b, c)



# Overlap calculation.
basisW = np.identity(nV) #Basis of the W state.
basisGHZ = np.array([np.zeros(nV),np.ones(nV)])
n = 1000  # Number of iterations of the overlap's sampling.
k = 5  # Number of iterations of Gibbs Sampling.

o = rbm_mag.Squared_Overlap(basisW, n, k)
print('O^2 of rbm_mag = ' + str(o))

# Print the amplitudes.
basis = rbm_mag.Basis(nV)
ampls =[]

for vec in basis:
    a = rbm_mag.waveF(vec)
    ampls.append(a)
print()
print('Amplitudes:')
for i in range(len(basis)):
    print(str(basis[i])+': '+str(ampls[i]))