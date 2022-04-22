# -*- coding: utf-8 -*-

from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator


def build(qc, ref_pos, n):
    qc = qc.copy()
    qc.h(ref_pos)
    # Add a CNOT between each qbit and the next:
    for i in range(ref_pos, n - 1):
        qc.cx(i, i + 1)
    return qc


# Testing the circuit.
# backend = AerSimulator()
# n = 10
# qc = QuantumCircuit(n)
# build(qc, n)
# qc.measure_all()
# display(qc.draw('mpl'))
# counts = backend.run(qc).result().get_counts()
# print(counts)
