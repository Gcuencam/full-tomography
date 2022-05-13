import itertools
from enum import Enum
import numpy as np
from src.common.quantum_commons import isDebugEnabled
from src.states.w import w_state_vector


class PauliBasis(Enum):
    X = 'X'
    Y = 'Y'
    Z = 'Z'


def measure_pauli(qc, measurement_schema):
    for i in range(qc.num_qubits):
        measurement = measurement_schema[i]
        debugMeasurement(measurement)
        if measurement == PauliBasis.X:
            x_measurement(qc, i, i)
        elif measurement == PauliBasis.Y:
            y_measurement(qc, i, i)
        elif measurement == PauliBasis.Z:
            z_measurement(qc, i, i)


def z_measurement(qc, qubit, cbit):
    """Measure 'qubit' in the X-basis, and store the result in 'cbit'"""
    qc.measure(qubit, cbit)
    return qc


def x_measurement(qc, qubit, cbit):
    """Measure 'qubit' in the X-basis, and store the result in 'cbit'"""
    qc.h(qubit)
    qc.measure(qubit, cbit)
    return qc


def y_measurement(qc, qubit, cbit):
    """Measure 'qubit' in the X-basis, and store the result in 'cbit'"""
    qc.sdg(qubit)
    qc.h(qubit)
    qc.measure(qubit, cbit)
    return qc


# Helper functions
def getCartesianPauliBasis(qc_size):
    basis = [PauliBasis.X, PauliBasis.Y, PauliBasis.Z]
    r = itertools.product(basis, repeat=qc_size)
    return np.array(list(r))


def debugMeasurement(measurement):
    if isDebugEnabled():
        print(measurement)


pauli_states = {
    PauliBasis.X: {
        '1': np.array([[1, 1]]).T,
        '-1': np.array([[1, -1]]).T
    },
    PauliBasis.Y: {
        '1': np.array([[1, 1j]]).T,
        '-1': np.array([[1, -1j]]).T
    },
    PauliBasis.Z: {
        '1': np.array([[1, 0]]).T,
        '-1': np.array([[0, 1]]).T
    }
}
pauli_projections = {
    PauliBasis.X: {
        '0': np.dot(pauli_states[PauliBasis.X]['1'], pauli_states[PauliBasis.X]['1'].T),
        '1': np.dot(pauli_states[PauliBasis.X]['-1'], pauli_states[PauliBasis.X]['-1'].T)
    },
    PauliBasis.Y: {
        '0': np.dot(pauli_states[PauliBasis.Y]['1'], pauli_states[PauliBasis.Y]['1'].T),
        '1': np.dot(pauli_states[PauliBasis.Y]['-1'], pauli_states[PauliBasis.Y]['-1'].T)
    },
    PauliBasis.Z: {
        '0': np.dot(pauli_states[PauliBasis.Z]['1'], pauli_states[PauliBasis.Z]['1'].T),
        '1': np.dot(pauli_states[PauliBasis.Z]['-1'], pauli_states[PauliBasis.Z]['-1'].T)
    }
}


def partial_least_square_estimator(schema, frequencies):
    q_size = len(schema)

    result = np.zeros((2 ** q_size))
    for i, state in enumerate(frequencies):
        kron = np.identity(1)
        for j, qbit_state in enumerate(state):
            qbit_basis = schema[j]
            pauli_projection = 3 * pauli_projections[qbit_basis][qbit_state] - np.identity(2)
            kron = np.kron(kron, pauli_projection)
        result = result + (frequencies[state] * kron)

    return result


def least_square_estimator(measurements):
    q_size = len(measurements[0])
    result = np.zeros((2 ** q_size))
    for measurement in measurements:
        schema = measurement['schema']
        schema_frequencies = measurement['frequencies']
        partial_sum = partial_least_square_estimator(schema, schema_frequencies)
        result = result + partial_sum

    return result / (3 ** q_size)


if __name__ == '__main__':
    qc_size = 2
    # Result with 1000000 shots
    measurements = [
        {'schema': [PauliBasis.X, PauliBasis.X], 'frequencies': {'11': 0.499732, '00': 0.500268}},
        {'schema': [PauliBasis.X, PauliBasis.Y], 'frequencies': {'01': 0.249122, '11': 0.250884, '00': 0.250373, '10': 0.249621}},
        {'schema': [PauliBasis.X, PauliBasis.Z], 'frequencies': {'01': 0.249319, '10': 0.24989, '00': 0.250122, '11': 0.250669}},
        {'schema': [PauliBasis.Y, PauliBasis.X], 'frequencies': {'11': 0.250305, '10': 0.249746, '00': 0.249751, '01': 0.250198}},
        {'schema': [PauliBasis.Y, PauliBasis.Y], 'frequencies': {'11': 0.499842, '00': 0.500158}},
        {'schema': [PauliBasis.Y, PauliBasis.Z], 'frequencies': {'11': 0.249783, '10': 0.250408, '00': 0.249202, '01': 0.250607}},
        {'schema': [PauliBasis.Z, PauliBasis.X], 'frequencies': {'11': 0.250426, '00': 0.250012, '10': 0.249633, '01': 0.249929}},
        {'schema': [PauliBasis.Z, PauliBasis.Y], 'frequencies': {'01': 0.250013, '10': 0.250427, '00': 0.250285, '11': 0.249275}},
        {'schema': [PauliBasis.Z, PauliBasis.Z], 'frequencies': {'10': 0.499241, '01': 0.500759}}
    ]

    rho_ls = least_square_estimator(measurements)
    print(rho_ls)
    overlap = np.sqrt(np.dot(w_state_vector, np.dot(rho_ls, w_state_vector.T)))
    print(overlap)
