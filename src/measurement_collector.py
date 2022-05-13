# -*- coding: utf-8 -*-

import numpy as np
from qiskit import QuantumCircuit

from .states.w import w_state_vector
from .common.quantum_commons import debug_circuit, isDebugEnabled, expandCounts, getFrequencies
from .common.quantum_commons import simulate
from .states.builder import build, States
from .measurements import pauli, sic_povm

def collect_pauli_measurements(type, qc_size, shots, output_filename):
    qc = QuantumCircuit(qc_size, qc_size)
    basis = pauli.getCartesianPauliBasis(qc_size)
    measurements = []
    ls_measurements = []

    for measurement_schema in basis:
        w_qc = build(type, qc, 0, qc_size)
        pauli.measure_pauli(w_qc, measurement_schema)
        job = simulate(w_qc, shots)
        counts = job.result().get_counts()
        measurements = expandCounts(counts)
        schema_frequencies = getFrequencies(counts, shots)
        ls_measurements.append({
            'schema': measurement_schema,
            'frequencies': schema_frequencies
        })

        if isDebugEnabled():
            debug_circuit(w_qc, counts, '')
    measurements = np.array(measurements)

    rho_ls = pauli.least_square_estimator(ls_measurements)
    overlap = np.dot(w_state_vector, np.dot(rho_ls, w_state_vector.T))[0][0]
    print(np.sqrt(overlap))

    if isDebugEnabled():
        print(measurements)
    print('Saving to ' + output_filename)
    np.save(output_filename, measurements)
    # np.savetxt(output_filename, measurements)
    # np.savetxt(output_filename, measurements, fmt='%s,', newline='\n', header='[', footer=']', comments='')


def collect_sic_povm_measurements(type, qc_size, shots, output_filename):
    qc = QuantumCircuit(qc_size, qc_size)
    qc = build(type, qc, 0, qc_size)
    # qc = plus_state()
    measured_circuit = sic_povm.measure_povm(qc)
    # measured_circuit = qc

    # Probabilities for measuring both qubits. In any measurement is applied in the circuit, this won't work.
    # psi = Statevector.from_instruction(measured_circuit)
    # probs = psi.probabilities_dict()
    # print('probs: {}'.format(probs))
    # print(np.array(psi))

    if isDebugEnabled():
        print(measured_circuit)

    job = simulate(measured_circuit, shots)
    counts = job.result().get_counts()
    measurements = expandCounts(counts)
    measurements = np.array(measurements)
    if isDebugEnabled():
        print(measurements)
    np.save(output_filename, measurements)
    # np.savetxt(output_filename, measurements)
    # np.savetxt(output_filename, measurements, fmt='%s,', newline='\n', header='[', footer=']', comments='')

    frequencies = getFrequencies(counts, shots)
    rho_ls = sic_povm.least_square_estimator(sic_povm.tetrahedron(), frequencies)

    overlap = np.dot(w_state_vector, np.dot(rho_ls, w_state_vector.T))[0][0]
    print(np.sqrt(overlap))


    # individualCounts = countPovmIndividual(counts)
    # print(individualCounts)

    # 2Qubits
    # print('Q0: 00: ' + str(povmCounts[0]['00'] / shots) + ', 01: ' + str(povmCounts[0]['01'] / shots) + ', 10: ' + str(povmCounts[0]['10'] / shots) + ', 11: ' + str(povmCounts[0]['11'] / shots))
    # print('Q1: 00: ' + str(povmCounts[2]['00'] / shots) + ', 01: ' + str(povmCounts[2]['01'] / shots) + ', 10: ' + str(povmCounts[2]['10'] / shots) + ', 11: ' + str(povmCounts[2]['11'] / shots))

    # 3Qubits
    # print('Q0: 00: ' + str(probs[0]['00']) + ', 01: ' + str(probs[0]['01']) + ', 10: ' + str(probs[0]['10']) + ', 11: ' + str(probs[0]['11']))
    # print('Q1: 00: ' + str(probs[2]['00']) + ', 01: ' + str(probs[2]['01']) + ', 10: ' + str(probs[2]['10']) + ', 11: ' + str(probs[2]['11']))
    # print('Q2: 00: ' + str(probs[4]['00']) + ', 01: ' + str(probs[4]['01']) + ', 10: ' + str(probs[4]['10']) + ', 11: ' + str(probs[4]['11']))



if __name__ == '__main__':
    qc_size = 3
    collect_pauli_measurements(States.GHZ, qc_size, 100000, 'training.npy')
