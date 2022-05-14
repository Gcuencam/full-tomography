from qiskit import QuantumCircuit
import numpy as np
import math
from qiskit.quantum_info import Statevector


def build(circuit, referencePosition, size):
    circuit.h(referencePosition)
    return circuit

def plus_state(size: int):
    circuit = QuantumCircuit(size, size)

    circuit.h(0)

    return circuit


def plus_state_rho():
    ket_plus_state = np.dot(1 / math.sqrt(2), np.array([[1, 1]]).T)
    bra_plus_state = np.dot(1 / math.sqrt(2), np.array([[1, 1]]))
    return np.dot(ket_plus_state, bra_plus_state)

plus_state_vector = np.array([math.sqrt(2), math.sqrt(2)]) / 2

if __name__ == '__main__':
    # Probabilities for measuring both qubits. In any measurement is applied in the circuit, this won't work.
    psi = Statevector.from_instruction(plus_state(1))
    probs = psi.probabilities_dict()
    print('probs: {}'.format(probs))
    print(np.array(psi))
    print(plus_state_vector)

