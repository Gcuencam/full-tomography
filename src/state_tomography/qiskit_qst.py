from qiskit import QuantumCircuit
from qiskit_experiments.library import StateTomography
from qiskit.providers.aer import AerSimulator
import numpy as np

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
    # print('probs: {}'.format(density_matrix.probabilities()))

def plus_state_tomography(shots):
    print('Plus state tomography')
    qc_size = 1
    circuit = QuantumCircuit(qc_size, qc_size)
    plus_qc = plus.build(circuit, 0, qc_size)
    density_matrix = tomography(plus_qc, shots)

    state_vector = get_plus_state_vector(qc_size)
    overlap = np.sqrt(np.dot(state_vector, np.dot(density_matrix, state_vector.T)))
    print(overlap)
    # print('probs: {}'.format(density_matrix.probabilities()))

def ghz_state_tomography(qc_size, shots):
    print('Plus state tomography')
    circuit = QuantumCircuit(qc_size, qc_size)
    ghz_qc = ghz.build(circuit, 0, qc_size)
    density_matrix = tomography(ghz_qc, shots)

    state_vector = get_ghz_state_vector(qc_size)
    overlap = np.sqrt(np.dot(state_vector, np.dot(density_matrix, state_vector.T)))
    print(overlap)
    # print('probs: {}'.format(density_matrix.probabilities()))

if __name__ == '__main__':
    qc_size = 4
    shots = 1000
    w_state_tomography(qc_size, shots)
    plus_state_tomography(shots)
    ghz_state_tomography(qc_size, shots)
