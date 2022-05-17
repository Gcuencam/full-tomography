import getopt
import json
import os
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import sys
import time
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit_experiments.library import StateTomography

from src.common.quantum_commons import simulate, getFrequencies
from src.measurements import sic_povm, pauli
from src.states.builder import build, get_state_vector, States


class ChartType(str, Enum):
    Overlaps = 'overlaps'
    Time = 'time'


def qiskit_tomography(experiment_state, qc_size, shots):
    qc = build(experiment_state, QuantumCircuit(qc_size, qc_size), 0, qc_size)
    # QST Experiment
    qstexp1 = StateTomography(qc)
    qstdata1 = qstexp1.run(AerSimulator(), seed_simulation=100, shots=shots).block_for_results()

    state_result = qstdata1.analysis_results("state")
    density_matrix = state_result.value
    return np.array(density_matrix)


def pauli_tomography(experiment_state, qc_size, shots):
    basis = pauli.getCartesianPauliBasis(qc_size)
    ls_measurements = []

    for measurement_schema in basis:
        qc = QuantumCircuit(qc_size, qc_size)
        w_qc = build(experiment_state, qc, 0, qc_size)
        w_qc = pauli.measure_pauli(w_qc, measurement_schema)
        job = simulate(w_qc, shots)
        counts = job.result().get_counts()
        schema_frequencies = getFrequencies(counts, shots)
        ls_measurements.append({
            'schema': measurement_schema,
            'frequencies': schema_frequencies
        })

    rho_ls = pauli.least_square_estimator(ls_measurements)
    return rho_ls


def sic_tomography(experiment_state, qc_size, shots):
    qc = QuantumCircuit(qc_size, qc_size)
    qc = build(experiment_state, qc, 0, qc_size)
    measured_circuit = sic_povm.measure_povm(qc)

    job = simulate(measured_circuit, shots)
    counts = job.result().get_counts()

    frequencies = getFrequencies(counts, shots)
    rho_ls = sic_povm.least_square_estimator(sic_povm.tetrahedron(), frequencies)

    return rho_ls


def append_result(results, experiment_state, qc_size, state_overlaps, time_profile):
    if experiment_state in results:
        results[experiment_state][qc_size] = {
            ChartType.Overlaps: state_overlaps,
            ChartType.Time: time_profile
        }
    else:
        results[experiment_state] = {
            qc_size: {
                ChartType.Overlaps: state_overlaps,
                ChartType.Time: time_profile
            }
        }


def generate_charts(results, chart_type, tomography_type, output_folder):
    folder = '%s/%s' % (output_folder, chart_type.value)
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i, state_key in enumerate(results):
        state = results[state_key]
        legend = []
        for j, qubit_size in enumerate(state):
            overlaps_by_qubit_size = state[qubit_size][chart_type]
            x_axis = overlaps_by_qubit_size.keys()
            y_axis = overlaps_by_qubit_size.values()
            plt.plot(x_axis, y_axis)
            legend.append('n = %s' % qubit_size)

        plt.legend(legend)
        plt.xlabel("N")
        if chart_type == ChartType.Overlaps:
            plt.ylabel("O")
        if chart_type == ChartType.Time:
            plt.ylabel("$t[ms]$")
        # plt.title(state_key)
        plt.savefig('%s/%s/%s_%s.png' % (output_folder, chart_type.value, state_key.value, tomography_type))
        plt.clf()


def perform_state_tomography(experiment_state, qc_size, shots, tomography_type):
    density_matrix = []
    if tomography_type == 'qiskit':
        density_matrix = qiskit_tomography(experiment_state, qc_size, shots)
    if tomography_type == 'pauli':
        density_matrix = pauli_tomography(experiment_state, qc_size, shots)
    if tomography_type == 'sic':
        density_matrix = sic_tomography(experiment_state, qc_size, shots)

    state_vector = get_state_vector(experiment_state, qc_size)
    overlap = abs(np.sqrt(np.dot(state_vector, np.dot(density_matrix, state_vector.T))))
    return overlap
    # print('probs: {}'.format(density_matrix.probabilities()))


def experiment(experiment_state, q_bits_range, shots_range, tomography_type, output_folder):
    results = {}
    for qc_size in q_bits_range:
        state_overlaps = {}
        state_time_profile = {}
        for shots in shots_range:
            print('Performing %s %s experiment with %s qubits and %s shots' % (tomography_type, experiment_state.value, qc_size, shots))
            ts = time.time()
            overlap = perform_state_tomography(experiment_state, qc_size, shots, tomography_type)
            print("Overlap: %s" % overlap)
            state_overlaps[shots] = overlap
            state_time_profile[shots] = time.time() - ts
            print('Time taken: %s in seconds' % state_time_profile[shots])

        append_result(results, experiment_state, qc_size, state_overlaps, state_time_profile)

    # Stores data in a file
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open('%s/%s_results.json' % (output_folder, experiment_state.value), "w") as fp:
        json.dump(results, fp)

    # Generate charts
    generate_charts(results, ChartType.Overlaps, tomography_type, output_folder)
    generate_charts(results, ChartType.Time, tomography_type, output_folder)


def main(argv):
    q_bits_lower_limit = 2
    q_bits_upper_limit = 6
    shots_lower_limit = 10
    shots_upper_limit = 100
    shots_pace = 10
    tomography_type = 'qiskit'
    output_folder = './'

    p_help = 'qiskit_qst.py --q_bits_lower_limit=<q_bits_lower_limit> --q_bits_upper_limit=<q_bits_upper_limit> --shots_lower_limit=<shots_lower_limit> --shots_upper_limit=<shots_upper_limit> --shots_pace=<shots_pace> --tomography_type=<tomography_type> --output_folder=<output_folder>'

    try:
        opts, args = getopt.getopt(argv, "", ["q_bits_lower_limit=", "q_bits_upper_limit=", "shots_lower_limit=",
                                              "shots_upper_limit=", "shots_pace=", "tomography_type=", "output_folder="])
    except getopt.GetoptError:
        print(p_help)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(p_help)
            sys.exit()
        elif opt == "--q_bits_lower_limit":
            q_bits_lower_limit = int(arg)
        elif opt == "--q_bits_upper_limit":
            q_bits_upper_limit = int(arg)
        elif opt == "--shots_lower_limit":
            shots_lower_limit = int(arg)
        elif opt == "--shots_upper_limit":
            shots_upper_limit = int(arg)
        elif opt == "--shots_pace":
            shots_pace = int(arg)
        elif opt == "--tomography_type":
            tomography_type = arg
        elif opt == "--output_folder":
            output_folder = arg

    q_bits_range = range(q_bits_lower_limit, q_bits_upper_limit + 1)
    shots_range = range(shots_lower_limit, shots_upper_limit + shots_pace, shots_pace)

    experiment(States.W, q_bits_range, shots_range, 'qiskit', output_folder)
    experiment(States.GHZ, q_bits_range, shots_range, 'qiskit', output_folder)
    # experiment(States.Plus, q_bits_range, shots_range, 'qiskit', output_folder)
    # experiment(States.W, q_bits_range, shots_range, 'pauli', output_folder)
    # experiment(States.GHZ, q_bits_range, shots_range, 'pauli', output_folder)
    # experiment(States.Plus, q_bits_range, shots_range, 'pauli', output_folder)
    # experiment(States.W, q_bits_range, shots_range, 'sic', output_folder)
    # experiment(States.GHZ, q_bits_range, shots_range, 'sic', output_folder)
    # experiment(States.Plus, q_bits_range, shots_range, 'sic', output_folder)


if __name__ == '__main__':
    main(sys.argv[1:])
