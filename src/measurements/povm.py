import itertools
import math

from qiskit import QuantumCircuit
from qiskit.circuit.library import Diagonal
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info.operators import Operator

import numpy as np

dividing = 12
alpha = math.sqrt((3 + math.sqrt(3)) / dividing)
beta = math.sqrt((3 - math.sqrt(3)) / dividing)

def get_m():
    a = math.sqrt(2) * alpha
    b = math.sqrt(2) * beta

    op = Operator([
        [a, b],
        [b, -a]
    ])
    return UnitaryGate(op, '~M')


def measure_qubit(qc, qubit, ancilla):
    qc.cx(qubit, ancilla)
    qc = qc.compose(get_m(), qubits=[ancilla])
    qc = qc.compose(Diagonal([1, 1, 1, 1j]), qubits=[ancilla, qubit])
    # qc = qc.compose(CPhaseGate(numpy.pi / 2), qubits=[ancilla, qubit])
    # qc = qc.compose(QFT(1), qubits=[qubit])
    qc.h(qubit)
    qc.measure(qubit, 2 * qubit)
    qc.measure(ancilla, (2 * qubit) + 1)

    return qc


def measure_povm(qc):
    size = qc.num_qubits

    expanded_circuit = QuantumCircuit(size * 2, size * 2)
    expanded_circuit = expanded_circuit.compose(qc, range(0, size))

    expanded_circuit.barrier(range(0, size))

    for i in range(qc.num_qubits):
        expanded_circuit = measure_qubit(expanded_circuit, i, i + size)

    return expanded_circuit

def tethrahedron():
    v_ket = [
        np.array([[alpha, beta]]).T,
        np.array([[alpha, -beta]]).T,
        np.array([[beta, 1j * alpha]]).T,
        np.array([[beta, -1j * alpha]]).T
    ]
    v_bra = [
        np.array([[alpha, beta]]),
        np.array([[alpha, -beta]]),
        np.array([[beta, -1j * alpha]]),
        np.array([[beta, 1j * alpha]])
    ]

    povm_size = len(v_ket)
    e = np.empty(povm_size, dtype=object)
    for i in range(0, povm_size):
        e[i] = np.dot(v_ket[i], v_bra[i])
    return {
        'kets': v_ket,
        'bras': v_bra,
        'e': e
    }


def get_cartessian_states(options, dimension):
    states = np.empty(2 ** len(options), dtype=object)
    cartesianProduct = list(itertools.product(options, repeat=dimension))
    for i, value in enumerate(cartesianProduct):
        states[i] = value

    return states


def pre_calculate_value(povm, dimension):
    possible_states = ['00', '01', '10', '11']
    povm_index_map = {
        '00': 0,
        '01': 1,
        '10': 2,
        '11': 3,
    }
    e = povm['e']

    states = get_cartessian_states(possible_states, dimension)
    result = {}
    for index, value in enumerate(states):
        m1 = (dimension + 1) * e[povm_index_map[value[0]]] - np.identity(dimension)
        m2 = (dimension + 1) * e[povm_index_map[value[1]]] - np.identity(dimension)
        # Not sure if I should use tensordot or kron
        # tensor = np.tensordot(m1, m2, axes=1)
        tensor = np.kron(m1, m2)
        result[value[0] + value[1]] = tensor

    return result

def least_square_estimator(frequencies, povm, dimension, n):
    pre_calculated_tensor = pre_calculate_value(povm, dimension)

    estimator = np.zeros((dimension ** 2, dimension ** 2))
    for index, state in enumerate(frequencies):
        fx = frequencies[state]
        pre_calculated_e = pre_calculated_tensor[state]
        estimator = estimator + (fx * pre_calculated_e)

    estimator = estimator / (dimension ** (n - 1))
    return estimator

# Test to get tr(Ex * rho)
def povm_test(rho):
    povm = tethrahedron()
    e = povm['e']
    povm_size = len(povm['kets'])

    result = np.empty(povm_size, dtype=object)
    for i in range(0, povm_size):
        r = np.dot(rho, e[i])
        trace = np.trace(r)
        result[i] = trace

    return result

if __name__ == '__main__':
    # qc_size = 2
    # w_qc = build(States.W, QuantumCircuit(qc_size, qc_size), 0, qc_size)
    # measured_circuit = measure_povm(w_qc)
    # print(measured_circuit)
    #
    # job = simulate(measured_circuit, 1000)
    # counts = job.result().get_counts()
    # print(counts)

    # Getting tr(Ex * rho) of plus state (|0> + |1>) / (m.sqrt(2)). rho == (<0| + <1|) / (m.sqrt(2))
    # print(povm_test(plus_state_rho()))

    # Testing least squares estimator
    frequencies = {'1001': 0.007, '1010': 0.028, '0101': 0.014, '1000': 0.056, '1110': 0.032, '0100': 0.178, '1100': 0.051, '1011': 0.035, '0110': 0.111, '0011': 0.09, '0001': 0.013, '1101': 0.003, '1111': 0.024, '0111': 0.101, '0010': 0.108, '0000': 0.149}
    dimension = 2
    n = 2
    r = least_square_estimator(frequencies, tethrahedron(), dimension, n)
    print(r)
