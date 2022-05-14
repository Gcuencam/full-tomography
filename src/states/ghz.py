# -*- coding: utf-8 -*-

from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
import numpy as np
import math as m
from qiskit.quantum_info import Statevector


def build(qc, ref_pos, n):
    qc = qc.copy()
    qc.h(ref_pos)
    # Add a CNOT between each qbit and the next:
    for i in range(ref_pos, n - 1):
        qc.cx(i, i + 1)
    return qc


ghz_state_vector = np.array([[m.sqrt(2), 0, 0, m.sqrt(2)]]) / 2
# [0.70710678+0.j 0.        +0.j 0.        +0.j 0.70710678+0.j]


if __name__ == '__main__':
    qc_size = 2
    circuit = QuantumCircuit(qc_size, qc_size)
    ghz_qc = build(circuit, 0, qc_size)
    psi = Statevector.from_instruction(ghz_qc)
    probs = psi.probabilities_dict()
    print('probs: {}'.format(probs))
    print(np.array(psi))
    print(ghz_state_vector)
