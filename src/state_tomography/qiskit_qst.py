import getopt

import matplotlib.pyplot as plt
import numpy as np
import sys
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit_experiments.library import StateTomography
from sys import path

path.append("../../")
from src.states import w, ghz, plus
from src.states.ghz import get_ghz_state_vector
from src.states.plus import get_plus_state_vector
from src.states.w import get_w_state_vector


def tomography(qc, shots):
    # QST Experiment
    qstexp1 = StateTomography(qc)
    qstdata1 = qstexp1.run(AerSimulator(), seed_simulation=100, shots=shots).block_for_results()

    state_result = qstdata1.analysis_results("state")
    density_matrix = state_result.value
    return density_matrix


def w_state_tomography(qc_size, shots):
    print('W state tomography')
    circuit = QuantumCircuit(qc_size, qc_size)
    w_qc = w.build(circuit, 0, qc_size)
    density_matrix = tomography(w_qc, shots)

    state_vector = get_w_state_vector(qc_size)
    overlap = np.sqrt(np.dot(state_vector, np.dot(density_matrix, state_vector.T)))
    print(overlap)
    return overlap
    # print('probs: {}'.format(density_matrix.probabilities()))


def plus_state_tomography(qc_size, shots):
    print('Plus state tomography')
    circuit = QuantumCircuit(qc_size, qc_size)
    plus_qc = plus.build(circuit, 0, qc_size)
    density_matrix = tomography(plus_qc, shots)

    state_vector = get_plus_state_vector(qc_size)
    overlap = np.sqrt(np.dot(state_vector, np.dot(density_matrix, state_vector.T)))
    print(overlap)
    return overlap
    # print('probs: {}'.format(density_matrix.probabilities()))


def ghz_state_tomography(qc_size, shots):
    print('Plus state tomography')
    circuit = QuantumCircuit(qc_size, qc_size)
    ghz_qc = ghz.build(circuit, 0, qc_size)
    density_matrix = tomography(ghz_qc, shots)

    state_vector = get_ghz_state_vector(qc_size)
    overlap = np.sqrt(np.dot(state_vector, np.dot(density_matrix, state_vector.T)))
    print(overlap)
    return overlap
    # print('probs: {}'.format(density_matrix.probabilities()))


def experiment(q_bits_range, shots_range):
    overlaps = {}
    for qc_size in q_bits_range:
        w_state_overlaps = {}
        ghz_state_overlaps = {}
        for shots in shots_range:
            w_state_overlaps[shots] = w_state_tomography(qc_size, shots)
            # plus_state_tomography(qc_size, shots)
            ghz_state_overlaps[shots] = ghz_state_tomography(qc_size, shots)

        if "w_state" in overlaps:
            overlaps["w_state"][qc_size] = w_state_overlaps
        else:
            overlaps["w_state"] = {
                qc_size: w_state_overlaps,
            }

        if "ghz_state" in overlaps:
            overlaps["ghz_state"][qc_size] = w_state_overlaps
        else:
            overlaps["ghz_state"] = {
                qc_size: w_state_overlaps,
            }

    for i, state_key in enumerate(overlaps):
        state = overlaps[state_key]
        for j, qubit_size in enumerate(state):
            title = '%s with %s qubits' % (state_key, qubit_size)
            plt.title(title)
            plt.xlabel("Shots")
            plt.ylabel("Overlap")
            overlaps_by_qubit_size = state[qubit_size]
            x_axis = overlaps_by_qubit_size.keys()
            y_axis = overlaps_by_qubit_size.values()
            plt.plot(x_axis, y_axis)
            plt.savefig('%s%s%s.png' % (i, j, title))
            plt.clf()

def main(argv):
    q_bits_lower_limit = 0
    q_bits_upper_limit = 6
    shots_lower_limit = 10
    shots_upper_limit = 100
    shots_pace = 10
    p_help = 'qiskit_qst.py --q_bits_lower_limit=<q_bits_lower_limit> --q_bits_upper_limit=<q_bits_upper_limit> --shots_lower_limit=<shots_lower_limit> --shots_upper_limit=<shots_upper_limit> --shots_pace=<shots_pace>'

    try:
        opts, args = getopt.getopt(argv, "", ["q_bits_lower_limit=", "q_bits_upper_limit=", "shots_lower_limit=", "shots_upper_limit=", "shots_pace="])
    except getopt.GetoptError:
        print(p_help)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(p_help)
            sys.exit()
        elif opt in ("-qll", "--q_bits_lower_limit"):
            q_bits_lower_limit = int(arg)
        elif opt in ("-qul", "--q_bits_upper_limit"):
            q_bits_upper_limit = int(arg)
        elif opt in ("-sll", "--shots_lower_limit"):
            shots_lower_limit = int(arg)
        elif opt in ("-sul", "--shots_upper_limit"):
            shots_upper_limit = int(arg)
        elif opt in ("-sp", "--shots_pace"):
            shots_pace = int(arg)

    q_bits_range = range(q_bits_lower_limit, q_bits_upper_limit)
    shots_range = range(shots_lower_limit, shots_upper_limit, shots_pace)

    experiment(q_bits_range, shots_range)

if __name__ == '__main__':
    main(sys.argv[1:])
