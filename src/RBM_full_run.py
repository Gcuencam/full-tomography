# -*- coding: utf-8 -*-
import numpy as np
import math
import time
from RBM_mag import RBM_mag
from RBM_phase import RBM_phase,TypeP,Param_idx
from configparser import ConfigParser


# Creates the network of magnitudes.
net_mag_file_path = 'rbm_mag.ini'
config = ConfigParser()
config.read(net_mag_file_path)
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


#Creates the network of phases.
net_phase_file_path = 'rbm_phase.ini'
config = ConfigParser()
config.read(net_phase_file_path)
nH = int(config.get('units', 'nH'))
_w = []
for i in range(nH):
    v = np.asarray(config.get('weights', str(i)).split(), dtype=np.float64)
    _w.append(v)
b = np.asarray(config.get('bias', 'b').split(), dtype=np.float64)
c = np.asarray(config.get('bias', 'c').split(), dtype=np.float64)
w = np.asarray(_w)

rbm_phase = RBM_phase(rbm_mag, nH, b, c, w)

#Print the phases of interest.
basisW = np.identity(nV)
for vec in basisW:
    print('Phase of ' + str(vec) + ': ' + str(rbm_phase.fi(vec)))
    
#Overlap.    
target_wf = np.array([0,(1/np.sqrt(3))*np.exp(complex(0,np.pi/4)),(1/np.sqrt(3))*np.exp(complex(0,np.pi/2)),0,(1/np.sqrt(3))*np.exp(complex(0,np.pi)),0,0,0])
o = rbm_phase.overlap(target_wf)
print('Overlap: ' + str(o))