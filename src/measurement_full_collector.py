# -*- coding: utf-8 -*-
import numpy as np
from qiskit import QuantumCircuit

from states.builder import build, States
from common.quantum_commons import simulate


def x_measurement(qc, qubit):
    """Measure 'qubit' in the X-basis, and store the result in 'cbit'"""
    qc.h(qubit)
    return qc


def y_measurement(qc, qubit):
    """Measure 'qubit' in the X-basis, and store the result in 'cbit'"""
    qc.sdg(qubit)
    qc.h(qubit)
    return qc


def stringToBitArray(str):
    arr = []
    for j in str:
        arr.append(int(j))
    return arr


def collect_measurements(type, qc_size, shots, output_filename):
    qc = QuantumCircuit(qc_size, qc_size)
    w_qc = build(type, qc, 0, qc_size)

    w_qc.barrier()

    measurements = []
    for i in range(shots):
        w_qc_copy = w_qc.copy()
        for j in range(qc_size):
            w_qc_copy.measure(j, j)

        job = simulate(w_qc_copy, 1)
        counts = job.result().get_counts()
        string = list(counts.keys())[0]
        measurements.append(stringToBitArray(string))
    np.save(output_filename + '_ZZ', measurements)

    for i in range(qc_size - 1):
        measurements = []
        for j in range(shots):
            w_qc_copy = w_qc.copy()
            x_measurement(w_qc_copy, i)
            x_measurement(w_qc_copy, i + 1)
            for k in range(qc_size):
                w_qc_copy.measure(k, k)
            job = simulate(w_qc_copy, 1)
            counts = job.result().get_counts()
            string = list(counts.keys())[0]
            measurements.append(stringToBitArray(string))
        np.save(output_filename + '_XX_' + str(i), measurements)

    for i in range(qc_size - 1):
        measurements = []
        for j in range(shots):
            w_qc_copy = w_qc.copy()
            x_measurement(w_qc_copy, i)
            y_measurement(w_qc_copy, i + 1)
            for k in range(qc_size):
                w_qc_copy.measure(k, k)
            job = simulate(w_qc_copy, 1)
            counts = job.result().get_counts()
            string = list(counts.keys())[0]
            measurements.append(stringToBitArray(string))
        np.save(output_filename + '_XY_' + str(i), measurements)


if __name__ == '__main__':
    qc_size = 3
    collect_measurements(States.W, qc_size, 10, './src/dataset/training_full')
