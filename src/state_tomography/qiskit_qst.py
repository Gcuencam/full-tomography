import qiskit
from qiskit import QuantumCircuit
from qiskit_experiments.library import StateTomography
from qiskit.providers.aer import QasmSimulator, AerSimulator

from src.states import w


def tomography(qc):
    # QST Experiment
    qstexp1 = StateTomography(qc)
    qstdata1 = qstexp1.run(AerSimulator(), seed_simulation=1000).block_for_results()

    state_result = qstdata1.analysis_results("state")
    density_matrix = state_result.value

    probs = density_matrix.probabilities()
    print(density_matrix)
    print('probs: {}'.format(probs))


if __name__ == '__main__':
    qc_size = 2
    circuit = QuantumCircuit(qc_size, qc_size)
    w_qc = w.build(circuit, 0, qc_size)
    tomography(w_qc)
