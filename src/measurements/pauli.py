import itertools
from enum import Enum
import numpy as np
from src.common.quantum_commons import isDebugEnabled


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
    basis = [PauliBasis.Z]
    r = itertools.product(basis, repeat=qc_size)
    return np.array(list(r))


def debugMeasurement(measurement):
    if isDebugEnabled():
        print(measurement)
