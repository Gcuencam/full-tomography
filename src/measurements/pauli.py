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
    tensor_frequencies = {}
    q_size = len(schema)

    for i, state in enumerate(frequencies):
        kron = np.empty((1, 1))
        for j, qbit_state in enumerate(state):
            qbit_basis = schema[j]
            pauli_projection = 3 * pauli_projections[qbit_basis][qbit_state] - np.identity(2)
            kron = np.kron(kron, pauli_projection)
        tensor_frequencies[state] = frequencies[state] * kron

    result = np.empty((2 ** q_size))
    for i, tensor_frequency in enumerate(tensor_frequencies):
        result = result + tensor_frequencies[tensor_frequency]

    return result


def least_square_estimator(measurements):
    q_size = len(measurements[0])
    result = np.empty((2 ** q_size))
    for measurement in measurements:
        schema = measurement['schema']
        schema_frequencies = measurement['frequencies']
        result = result + partial_least_square_estimator(schema, schema_frequencies)

    return result / (3 ** q_size)


if __name__ == '__main__':
    qc_size = 2
    settings = getCartesianPauliBasis(qc_size)
    schema = settings[0]
    schema_frequencies = {
        '00': 0.25,
        '01': 0.25,
        '10': 0.25,
        '11': 0.25
    }
    measurements = []
    for setting in settings:
        measurements.append({
            'schema': setting,
            'frequencies': schema_frequencies
        })

    rho_ls = least_square_estimator(measurements)
    print(rho_ls)
    overlap = np.sqrt(np.dot(w_state_vector, np.dot(rho_ls, w_state_vector.T)))
    print(overlap)
