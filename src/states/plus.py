from qiskit import QuantumCircuit
import numpy as np
import math


def plus_state(size: int):
    circuit = QuantumCircuit(size, size)

    circuit.h(0)

    return circuit


def plus_state_rho():
    ket_plus_state = np.dot(1 / math.sqrt(2), np.array([[1, 1]]).T)
    bra_plus_state = np.dot(1 / math.sqrt(2), np.array([[1, 1]]))
    return np.dot(ket_plus_state, bra_plus_state)

