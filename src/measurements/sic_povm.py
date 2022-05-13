import math

from qiskit import QuantumCircuit
from qiskit.circuit.library import Diagonal
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info.operators import Operator

import numpy as np

from src.states.plus import plus_state_rho
from src.states.w import w_state_vector

divisor = 12
alpha = math.sqrt((3 + math.sqrt(3)) / divisor)
beta = math.sqrt((3 - math.sqrt(3)) / divisor)


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


def tetrahedron():
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
        'e': e,
        'e_states': {
            '00': e[0],
            '01': e[1],
            '10': e[2],
            '11': e[3]
        }
    }


def least_square_estimator(povm, frequencies):
    e_states = povm['e_states']
    povm_dimension = list(povm['e_states'].values())[0].shape[0]
    n_qubits = len(list(frequencies.keys())[0]) / povm_dimension

    result = np.zeros((povm_dimension ** 2))
    for i, state in enumerate(frequencies):
        kron = np.identity(1)
        for j in range(0, len(state), 2):
            qbit_state = state[j] + state[j + 1]
            povm_i = (povm_dimension + 1) * e_states[qbit_state] - np.identity(povm_dimension)
            kron = np.kron(kron, povm_i)
        result = result + (frequencies[state] * kron)

    return result / (povm_dimension ** (n_qubits - 1))


# Test to get tr(Ex * rho)
def tetrahedron_test(rho):
    povm = tetrahedron()
    e = povm['e']
    povm_size = len(povm['kets'])

    result = np.empty(povm_size, dtype=object)
    for i in range(0, povm_size):
        r = np.dot(rho, e[i])
        trace = np.trace(r)
        result[i] = trace

    return result


def least_squares_test():
    # Testing least squares estimator
    w_state_frequencies = {
        '0000': 0.0845,
        # '0001': 0,
        '0010': 0.0842,
        '0011': 0.0839,
        # '0100': 0,
        '0101': 0.0818,
        '0110': 0.0829,
        '0111': 0.0782,
        '1000': 0.0876,
        '1001': 0.0851,
        '1010': 0.0802,
        # '1011': 0,
        '1100': 0.0837,
        '1101': 0.0843,
        # '1110': 0,
        '1111': 0.0836
    }
    rho_ls = least_square_estimator(tetrahedron(), w_state_frequencies)
    return np.sqrt(np.dot(w_state_vector, np.dot(rho_ls, w_state_vector.T))[0][0])


if __name__ == '__main__':
    # qc_size = 2
    # w_qc = build(States.W, QuantumCircuit(qc_size, qc_size), 0, qc_size)
    # measured_circuit = measure_povm(w_qc)
    # print(measured_circuit)
    #
    # job = simulate(measured_circuit, 1000)
    # counts = job.result().get_counts()
    # print(counts)

    # Getting tr(Ex * rho) of plus state (|0> + |1>) / (m.sqrt(2)).
    print(tetrahedron_test(plus_state_rho()))
    print(least_squares_test())
