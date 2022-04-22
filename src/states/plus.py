from qiskit import QuantumCircuit

def plus_state(size: int):
    circuit = QuantumCircuit(size, size)

    circuit.h(0)

    return circuit