# -*- coding: utf-8 -*-

from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
backend = AerSimulator()

n = 10

qc = QuantumCircuit(n)
qc.h(0)
#Add a CNOT between each qbit and the next: 
for i in range(n-1):
    qc.cx(i,i+1)
    

qc.measure_all()
display(qc.draw('mpl'))
counts = backend.run(qc).result().get_counts()
print(counts)

