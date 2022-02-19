import math as m
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.quantum_info import Statevector


def w_state(n: int):
    circuit = QuantumCircuit(n, n)

    if n < 0:
        raise Exception('The size must be at least two.')
    if n == 2:
        circuit.h(0)
    else:
        circuit.ry(2 * np.arccos(1 / m.sqrt(n)), 0)

        for i in range(1, n - 2):
            circuit.cry(2 * np.arccos(1 / m.sqrt(n - i)), i - 1, i)

        circuit.ch(n - 3, n - 2)

    for i in range(n - 1, 0, -1):
        circuit.cx(i - 1, i)

    circuit.x(0)
    return circuit


def compose_w_state(qc, q_reference, q_size):
    if q_reference < 0:
        raise Exception('The reference position must be at least zero.')
    if q_reference + q_size > qc.num_qubits:
        raise Exception('The circuit does not have enough qubits.')
    wqc = w_state(q_size)
    return qc.compose(wqc, qubits=range(q_reference, q_reference + q_size))


def measure(qc):
    for i in range(1, qc.num_qubits):
        qc.measure(i, i)


def testWState():
    simulator = QasmSimulator()

    qc_size = 4
    w_state_initial_qubit_position = 1
    w_state_size = 3
    composed_qc = compose_w_state(QuantumCircuit(qc_size, qc_size), w_state_initial_qubit_position, w_state_size)

    psi = Statevector.from_instruction(composed_qc)
    probs = psi.probabilities_dict()

    measure(composed_qc)

    compiled_circuit = transpile(composed_qc, simulator)
    shots = 10
    job = simulator.run(compiled_circuit, shots=shots)

    counts = job.result().get_counts()

    print(composed_qc)
    print(counts)
    print('probs: {}'.format(probs))
    assert len(probs) == w_state_size
    offset = 0.02
    upper_bound = (shots / w_state_size / shots) + offset
    lower_bound = (shots / w_state_size / shots) - offset
    for key, value in probs.items():
        assert value < upper_bound
        assert value > lower_bound


if __name__ == '__main__':
    testWState()
