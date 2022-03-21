# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 19:40:28 2022

@author: Pedro
"""

import itertools
from enum import Enum
import numpy as np
from quantum_commons import debug_circuit, isDebugEnabled
from quantum_commons import simulate
from w_state import w_state


class PauliBasis(Enum):
    X = 'X'
    Y = 'Y'
    Z = 'Z'


def collect_measurements(qc_size, shots, output_filename):
    basis = getCartesianPauliBasis(qc_size)
    measurements = []

    for measurement_schema in basis:
        w_qc = w_state(qc_size)
        measure(w_qc, measurement_schema)
        job = simulate(w_qc, shots)
        counts = job.result().get_counts()
        get_measurements(measurements, counts)

        if isDebugEnabled():
            debug_circuit(w_qc, counts, '')
    measurements = np.array(measurements)
    
    if isDebugEnabled():
        print(measurements)
    print('Saving to ' + output_filename)
    np.save(output_filename, measurements)

def get_measurements(measurements, counts):
    meas_out = list(counts.keys())
    for meas in meas_out:
        measurement = []
        for bit in meas:
            measurement.append(int(bit))
        for i in range(counts[meas]):
            measurements.append(measurement.copy())
        

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
    if isDebugEnabled():
        print(measurement)


if __name__ == '__main__':
    qc_size = 3
    collect_measurements(qc_size, 100000, 'training.npy')
