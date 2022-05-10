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
    r = range(qc_size)
    qc.measure(r, r)
    print(qc.draw())
    job = simulate(qc, shots)
    counts = job.result().get_counts()
    
    measurements = []
    meas_out = list(counts.keys())
    for meas in meas_out:   
        measurement = int(meas,2)
        for i in range(counts[meas]):
            measurements.append(measurement)
    np.random.shuffle(measurements)
    return measurements


def collect_measurements(type, qc_size, shots, output_filename):
    qc = QuantumCircuit(qc_size, qc_size)
    w_qc = build(type, qc, 0, qc_size)
    w_qc.barrier()
    w_qc = changePhase(w_qc, [np.pi/4, np.pi/2, np.pi])
    w_qc.barrier()

    #Z measurements
    w_qc_z = w_qc.copy()
    r = range(qc_size)
    w_qc_z.measure(r, r)
    print(w_qc_z.draw())
    job = simulate(w_qc_z, shots)
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
    np.savetxt(output_filename + '_Z' + '.txt', measurements)

    #XX measurements
    j = 0
    for i in range(qc_size-1,0,-1):
        w_qc_xx = w_qc.copy()
        x_measurement(w_qc_xx, i)
        x_measurement(w_qc_xx, i - 1)
        w_qc_xx.barrier()
        measurements = get_measurements(w_qc_xx, qc_size, shots)
        np.savetxt(output_filename + '_XX_' + str(j) + '.txt', measurements)
        j+=1

    #XY measurements
    j = 0
    for i in range(qc_size-1,0,-1):
        w_qc_xy = w_qc.copy()
        x_measurement(w_qc_xy, i)
        y_measurement(w_qc_xy, i - 1)
        w_qc_xy.barrier()
        measurements = get_measurements(w_qc_xy, qc_size, shots)
        np.savetxt(output_filename + '_XY_' + str(j) + '.txt', measurements)
        j+=1


if __name__ == '__main__':
    
    qc_size = 3
    collect_measurements(States.W, qc_size, 5000, 'dataset/training_full')