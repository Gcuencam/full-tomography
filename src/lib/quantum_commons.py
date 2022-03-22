# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 19:42:48 2022

@author: Pedro
"""

import os

from qiskit import transpile
from qiskit.providers.aer import QasmSimulator


def simulate(qc, shots):
    simulator = QasmSimulator()
    compiled_circuit = transpile(qc, simulator)
    return simulator.run(compiled_circuit, shots=shots)


def debug_circuit(qc, counts, probs):
    print(qc)
    print(counts)
    print('probs: {}'.format(probs))


def isDebugEnabled():
    if "DEBUG" in os.environ:
        return os.environ['DEBUG'] == 'True'
    else:
        return False