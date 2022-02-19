import itertools
from enum import Enum

import numpy as np

from commons import debug
from commons import simulate
from w_state import w_state


class PauliBasis(Enum):
    X = 'X'
    Y = 'Y'
    Z = 'Z'


def collect_measurements(qc_size):
    basis = getCartesianPauliBasis(qc_size)
    result = np.empty(0)

    for measurement_schema in basis:
        w_qc = w_state(qc_size)
        measure(w_qc, measurement_schema)
        shots = 1
        job = simulate(w_qc, shots)
        counts = job.result().get_counts()

        result = np.append(result, list(counts))
        debug(w_qc, counts, '')

    print(result)


def measure(qc, measurement_schema):
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
    print(measurement)


if __name__ == '__main__':
    qc_size = 3
    collect_measurements(qc_size)
