# -*- coding: utf-8 -*-

import itertools
from enum import Enum
import numpy as np
from qiskit.quantum_info import Statevector

from .quantum_commons import debug_circuit, isDebugEnabled
from .quantum_commons import simulate
from .measurements.povm import measure_povm
from .w_state import w_state


class PauliBasis(Enum):
    X = 'X'
    Y = 'Y'
    Z = 'Z'


def collect_pauli_measurements(qc_size, shots, output_filename):
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
    # np.savetxt(output_filename, measurements)
    # np.savetxt(output_filename, measurements, fmt='%s,', newline='\n', header='[', footer=']', comments='')


def collect_povm_measurements(qc_size, shots, output_filename):
    measured_circuit = w_state(qc_size)
    measured_circuit = measure_povm(measured_circuit)

    # Probabilities for measuring both qubits. In any measurement is applied in the circuit, this won't work.
    # psi = Statevector.from_instruction(measured_circuit)
    # probs = psi.probabilities_dict()
    # print('probs: {}'.format(probs))

    measurements = []
    if isDebugEnabled():
        print(measured_circuit)

    job = simulate(measured_circuit, shots)
    counts = job.result().get_counts()

    get_measurements(measurements, counts)
    # if isDebugEnabled():
    #     print(measurements)
    # np.savetxt(output_filename, measurements)
    np.savetxt(output_filename, measurements, fmt='%s,', newline='\n', header='[', footer=']', comments='')

    counts = countPovm(measurements)
    probs = getProbabilities(counts, shots)
    print(probs)

    #2Qubits
    # print('Q0: 00: ' + str(probs[0]['00']) + ', 01: ' + str(probs[0]['01']) + ', 10: ' + str(probs[0]['10']) + ', 11: ' + str(probs[0]['11']))
    # print('Q1: 00: ' + str(probs[2]['00']) + ', 01: ' + str(probs[2]['01']) + ', 10: ' + str(probs[2]['10']) + ', 11: ' + str(probs[2]['11']))

    #3Qubits
    # print('Q0: 00: ' + str(probs[0]['00']) + ', 01: ' + str(probs[0]['01']) + ', 10: ' + str(probs[0]['10']) + ', 11: ' + str(probs[0]['11']))
    # print('Q1: 00: ' + str(probs[2]['00']) + ', 01: ' + str(probs[2]['01']) + ', 10: ' + str(probs[2]['10']) + ', 11: ' + str(probs[2]['11']))
    # print('Q2: 00: ' + str(probs[4]['00']) + ', 01: ' + str(probs[4]['01']) + ', 10: ' + str(probs[4]['10']) + ', 11: ' + str(probs[4]['11']))


def countPovm(measurements):
    qbits_count = {}
    for bits in measurements:
        for q_index in range(0, len(bits), 2):
            if q_index in qbits_count:
                q_counts = qbits_count[q_index]
                q_value = str(bits[q_index]) + str(bits[q_index + 1])
                if q_value in q_counts:
                    q_counts[q_value] = q_counts[q_value] + 1
                else:
                    q_counts[q_value] = 1
            else:
                q_counts = {}
                qbits_count[q_index] = q_counts
                q_value = str(bits[q_index]) + str(bits[q_index + 1])
                if q_value in q_counts:
                    q_counts[q_value] = q_counts[q_value] + 1
                else:
                    q_counts[q_value] = 1

    return qbits_count


def getProbabilities(qbits_count, shots):
    probs = {}
    for q_index in qbits_count:
        q_bit_count = qbits_count[q_index]
        for q_bit_measurement in q_bit_count:
            counts = q_bit_count[q_bit_measurement]
            if q_index in probs:
                probs[q_index][q_bit_measurement] = counts / shots
            else:
                probs[q_index] = {}
                probs[q_index][q_bit_measurement] = counts / shots
    return probs


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
    collect_pauli_measurements(qc_size, 64, 'test.txt')
