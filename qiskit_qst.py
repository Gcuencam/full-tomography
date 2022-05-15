import getopt
import time
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import sys
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit_experiments.library import StateTomography
from src.common.quantum_commons import simulate, getFrequencies
from src.measurements import sic_povm, pauli
from src.states.builder import build, get_state_vector, States


class ChartType(Enum):
    Overlaps = 'overlaps'
    Time = 'time'


def qiskit_tomography(type, qc_size, shots):
    qc = build(type, QuantumCircuit(qc_size, qc_size), 0, qc_size)
    # QST Experiment
    qstexp1 = StateTomography(qc)
    qstdata1 = qstexp1.run(AerSimulator(), seed_simulation=100, shots=shots).block_for_results()

    state_result = qstdata1.analysis_results("state")
    density_matrix = state_result.value
    return density_matrix


def pauli_tomography(type, qc_size, shots):
    qc = QuantumCircuit(qc_size, qc_size)
    basis = pauli.getCartesianPauliBasis(qc_size)
    ls_measurements = []

    for measurement_schema in basis:
        w_qc = build(type, qc, 0, qc_size)
        pauli.measure_pauli(w_qc, measurement_schema)
        job = simulate(w_qc, shots)
        counts = job.result().get_counts()
        schema_frequencies = getFrequencies(counts, shots)
        ls_measurements.append({
            'schema': measurement_schema,
            'frequencies': schema_frequencies
        })

    rho_ls = pauli.least_square_estimator(ls_measurements)
    return rho_ls

def sic_tomography(type, qc_size, shots):
    qc = QuantumCircuit(qc_size, qc_size)
    qc = build(type, qc, 0, qc_size)
    measured_circuit = sic_povm.measure_povm(qc)

    job = simulate(measured_circuit, shots)
    counts = job.result().get_counts()

    frequencies = getFrequencies(counts, shots)
    rho_ls = sic_povm.least_square_estimator(sic_povm.tetrahedron(), frequencies)

    return rho_ls


def append_result(results, experiment_state, qc_size, state_overlaps, time_profile, ):
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


def generate_charts(results, chart_type):
    for i, state_key in enumerate(results):
        state = results[state_key]
        legend = []
        for j, qubit_size in enumerate(state):
            overlaps_by_qubit_size = state[qubit_size][chart_type]
            x_axis = overlaps_by_qubit_size.keys()
            y_axis = overlaps_by_qubit_size.values()
            plt.plot(x_axis, y_axis)
            legend.append('%s qubits' % qubit_size)

        title = state_key
        plt.legend(legend)
        plt.xlabel("N")
        plt.ylabel(chart_type)
        # plt.title(title)
        plt.savefig('%s_chart_%s.png' % (chart_type, title))
        plt.clf()


def perform_state_tomography(type, qc_size, shots, tomography_type):
    density_matrix = []
    if tomography_type == 'qiskit':
        density_matrix = qiskit_tomography(type, qc_size, shots)
    if tomography_type == 'pauli':
        density_matrix = pauli_tomography(type, qc_size, shots)
    if tomography_type == 'sic':
        density_matrix = sic_tomography(type, qc_size, shots)

    state_vector = get_state_vector(type, qc_size)
    overlap = np.sqrt(np.dot(state_vector, np.dot(density_matrix, state_vector.T)))
    print(overlap)
    return overlap
    # print('probs: {}'.format(density_matrix.probabilities()))


def experiment(q_bits_range, shots_range, tomography_type):
    results = {}
    for qc_size in q_bits_range:
        w_state_overlaps = {}
        w_state_time_profile = {}
        ghz_state_overlaps = {}
        ghz_state_time_profile = {}
        for shots in shots_range:
            print('Performing %s experiment with %s qubits and %s shots' % (States.W, qc_size, shots))
            ts = time.time()
            w_state_overlaps[shots] = perform_state_tomography(States.W, qc_size, shots, tomography_type)
            w_state_time_profile[shots] = time.time() - ts

            ts = time.time()
            ghz_state_overlaps[shots] = perform_state_tomography(States.GHZ, qc_size, shots, tomography_type)
            ghz_state_time_profile[shots] = time.time() - ts

        append_result(results, States.W, qc_size, w_state_overlaps, w_state_time_profile)
        append_result(results, States.GHZ, qc_size, ghz_state_overlaps, ghz_state_time_profile)

    generate_charts(results, ChartType.Overlaps)
    generate_charts(results, ChartType.Time)


def main(argv):
    q_bits_lower_limit = 0
    q_bits_upper_limit = 6
    shots_lower_limit = 10
    shots_upper_limit = 100
    shots_pace = 10
    tomography_type = 'qiskit'
    p_help = 'qiskit_qst.py --q_bits_lower_limit=<q_bits_lower_limit> --q_bits_upper_limit=<q_bits_upper_limit> --shots_lower_limit=<shots_lower_limit> --shots_upper_limit=<shots_upper_limit> --shots_pace=<shots_pace> --tomography_type=<tomography_type>'

    try:
        opts, args = getopt.getopt(argv, "", ["q_bits_lower_limit=", "q_bits_upper_limit=", "shots_lower_limit=", "shots_upper_limit=", "shots_pace=", "tomography_type="])
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

    q_bits_range = range(q_bits_lower_limit, q_bits_upper_limit + 1)
    shots_range = range(shots_lower_limit, shots_upper_limit + shots_pace, shots_pace)

    experiment(q_bits_range, shots_range, tomography_type)


if __name__ == '__main__':
    main(sys.argv[1:])
