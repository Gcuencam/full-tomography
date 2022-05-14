from qiskit import QuantumCircuit
from qiskit_experiments.library import StateTomography
from qiskit.providers.aer import AerSimulator
import numpy as np
from src.states import w, ghz, plus
from src.states.ghz import ghz_state_vector
from src.states.plus import plus_state_vector
from src.states.w import w_state_vector


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
    probs = density_matrix.probabilities()

    overlap = np.sqrt(np.dot(w_state_vector, np.dot(density_matrix, w_state_vector.T)))
    print(overlap)
    # print('probs: {}'.format(probs))

def plus_state_tomography(shots):
    print('Plus state tomography')
    qc_size = 1
    circuit = QuantumCircuit(qc_size, qc_size)
    plus_qc = plus.build(circuit, 0, qc_size)
    density_matrix = tomography(plus_qc, shots)
    probs = density_matrix.probabilities()

    overlap = np.sqrt(np.dot(plus_state_vector, np.dot(density_matrix, plus_state_vector.T)))
    print(overlap)
    # print('probs: {}'.format(probs))

def ghz_state_tomography(qc_size, shots):
    print('Plus state tomography')
    circuit = QuantumCircuit(qc_size, qc_size)
    ghz_qc = ghz.build(circuit, 0, qc_size)
    density_matrix = tomography(ghz_qc, shots)
    probs = density_matrix.probabilities()

    overlap = np.sqrt(np.dot(ghz_state_vector, np.dot(density_matrix, ghz_state_vector.T)))
    print(overlap)
    # print('probs: {}'.format(probs))

if __name__ == '__main__':
    qc_size = 2
    shots = 1000
    w_state_tomography(qc_size, shots)
    plus_state_tomography(shots)
    ghz_state_tomography(qc_size, shots)
