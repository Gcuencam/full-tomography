# -*- coding: utf-8 -*-
import numpy as np
import math
import time
from RBM_mag import RBM_mag
from RBM_phase import RBM_phase,TypeP,Param_idx
from configparser import ConfigParser


# Creates the network of magnitudes.
net_mag_path = 'rbm_mag.ini'
config = ConfigParser()
config.read(net_mag_path)
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



#Trains the network of phases.
nH = 3
epochs = 20
data_path = 'dataset/training_full_'
net_phase_path = 'rbm_phase.ini'
batch_size = 10
learnRate = 0.1

rbm_phase = RBM_phase(rbm_mag,nH)

t = time.time()
rbm_phase.train(epochs, data_path, batch_size, learnRate)
print('Elapsed time:', time.time() - t)

    
#Saves the network of phases.
config = ConfigParser()
config.add_section('units')
config.set('units', 'nH', str(rbm_phase.nH))
config.add_section('weights')
for i, v in enumerate(rbm_phase.w):
    config.set('weights', str(i), ' '.join(map(str, v)))
config.add_section('bias')
config.set('bias', 'b', ' '.join(map(str, rbm_phase.b)))
config.set('bias', 'c', ' '.join(map(str, rbm_phase.c)))
with open(net_phase_path, 'w') as f:
    config.write(f)
