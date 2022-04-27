# -*- coding: utf-8 -*-
import numpy as np
from qiskit import QuantumCircuit

from states.builder import build, States
from common.quantum_commons import simulate


def x_measurement(qc, qubit):
    """Measure 'qubit' in the X-basis, and store the result in 'cbit'"""
    qc.h(qubit)



def y_measurement(qc, qubit):
    """Measure 'qubit' in the X-basis, and store the result in 'cbit'"""
    qc.sdg(qubit)
    qc.h(qubit)


def changePhase(circuit, phases):
    for p in phases:
        if p < 0 or p > 2 * np.pi:
            raise Exception('The phases must be between 0 and 2pi.')

    _circuit = circuit.copy()
    for i in range(len(phases)):
        if phases[i] != 0:
            _circuit.p(phases[i], i)
    return _circuit


def get_measurements(qc, qc_size, shots):
    for i in range(qc_size):
        qc.measure(i, i)
    print(qc.draw())
    job = simulate(qc, shots)
    counts = job.result().get_counts()
    
    measurements = []
    meas_out = list(counts.keys())
    for meas in meas_out:
        measurement = []
        for bit in meas:
            measurement.append(int(bit))
        for i in range(counts[meas]):
            measurements.append(measurement.copy())
    np.random.shuffle(measurements)
    return measurements


def collect_measurements(type, qc_size, shots, output_filename):
    qc = QuantumCircuit(qc_size, qc_size)
    w_qc = build(type, qc, 0, qc_size)
    w_qc.barrier()
    w_qc = changePhase(w_qc, [0, 0, 0])
    w_qc.barrier()


    #Z measurements
    w_qc_z = w_qc.copy()
    measurements = get_measurements(w_qc_z, qc_size, shots)
    np.savetxt(output_filename + '_Z.txt', measurements)

    #XX measurements
    for i in range(qc_size - 1):
        w_qc_xx = w_qc.copy()
        x_measurement(w_qc_xx, i)
        x_measurement(w_qc_xx, i + 1)
        w_qc_xx.barrier()
        measurements = get_measurements(w_qc_xx, qc_size, shots)
        np.savetxt(output_filename + '_XX_' + str(i) + '.txt', measurements)


    #XY measurements
    for i in range(qc_size - 1):
        w_qc_xy = w_qc.copy()
        x_measurement(w_qc_xy, i)
        y_measurement(w_qc_xy, i + 1)
        w_qc_xy.barrier()
        measurements = get_measurements(w_qc_xy, qc_size, shots)
        np.savetxt(output_filename + '_XY_' + str(i) + '.txt', measurements)


if __name__ == '__main__':
    
    qc_size = 3
    collect_measurements(States.W, qc_size, 10000, './src/dataset/training_full')