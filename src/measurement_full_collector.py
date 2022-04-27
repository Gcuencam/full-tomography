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


def changePhase(circuit, phases):
    for p in phases:
        if p < 0 or p > 2 * np.pi:
            raise Exception('The phases must be between 0 and 2pi.')

    _circuit = circuit.copy()
    for i in range(len(phases)):
        if phases[i] != 0:
            _circuit.p(phases[i], i)
    return _circuit


def stringToBitArray(str):
    arr = []
    for j in str:
        arr.append(int(j))
    return arr


def collect_measurements(type, qc_size, shots, output_filename):
    qc = QuantumCircuit(qc_size, qc_size)
    w_qc = build(type, qc, 0, qc_size)
    w_qc.barrier()
    w_qc = changePhase(w_qc, [0, 0, 0])
    w_qc.barrier()

    #Z measurements.
    measurements = []
    for i in range(shots):
        w_qc_copy = w_qc.copy()
        for j in range(qc_size):
            w_qc_copy.measure(j, j)
        job = simulate(w_qc_copy, 1)
        counts = job.result().get_counts()
        string = list(counts.keys())[0]
        measurements.append(stringToBitArray(string))
    np.savetxt(output_filename + '_Z.txt', measurements)

    #XX measurements.
    for i in range(qc_size - 1):
        measurements = []
        w_qc_xx = w_qc.copy()
        x_measurement(w_qc_xx, i)
        x_measurement(w_qc_xx, i + 1)
        w_qc_xx.barrier()
        for k in range(qc_size):
            w_qc_xx.measure(k, k)
        for j in range(shots):
            w_qc_copy = w_qc_xx.copy()
            job = simulate(w_qc_copy, 1)
            counts = job.result().get_counts()
            string = list(counts.keys())[0]
            measurements.append(stringToBitArray(string))       
        np.savetxt(output_filename + '_XX_' + str(i) + '.txt', measurements)

    #XY measurements.
    for i in range(qc_size - 1):
        measurements = []
        w_qc_xy = w_qc.copy()
        x_measurement(w_qc_xy, i)
        y_measurement(w_qc_xy, i + 1)
        w_qc_xy.barrier()
        for k in range(qc_size):
            w_qc_xy.measure(k, k)
        for j in range(shots):
            w_qc_copy = w_qc_xy.copy()
            job = simulate(w_qc_copy, 1)
            counts = job.result().get_counts()
            string = list(counts.keys())[0]
            measurements.append(stringToBitArray(string))
        np.savetxt(output_filename + '_XY_' + str(i) + '.txt', measurements)


if __name__ == '__main__':
    qc_size = 3
    collect_measurements(States.W, qc_size, 10, './src/dataset/training_full')