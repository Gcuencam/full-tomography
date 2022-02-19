import math as m

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from commons import debug
from commons import simulate


def w_state(size: int):
    circuit = QuantumCircuit(size, size)

    if size < 0:
        raise Exception('The size must be at least two.')
    if size == 2:
        circuit.h(0)
    else:
        circuit.ry(2 * np.arccos(1 / m.sqrt(size)), 0)

        for target_qubit in range(1, size - 2):
            control_qubit = target_qubit - 1
            circuit.cry(2 * np.arccos(1 / m.sqrt(size - target_qubit)), control_qubit, target_qubit)

        circuit.ch(size - 3, size - 2)

    for target_qubit in range(size - 1, 0, -1):
        control_qubit = target_qubit - 1
        circuit.cx(control_qubit, target_qubit)

    circuit.x(0)
    return circuit


def compose_w_state(qc, q_reference, q_size):
    if q_reference < 0:
        raise Exception('The reference position must be at least zero.')
    if q_reference + q_size > qc.num_qubits:
        raise Exception('The circuit does not have enough qubits.')
    w_qc = w_state(q_size)
    return qc.compose(w_qc, qubits=range(q_reference, q_reference + q_size))


def testWState(qc_size):
    w_state_initial_qubit_position = 1
    w_state_size = 3
    qc = compose_w_state(QuantumCircuit(qc_size, qc_size), w_state_initial_qubit_position, w_state_size)

    psi = Statevector.from_instruction(qc)
    probs = psi.probabilities_dict()

    qc.measure(1, 1)
    qc.measure(2, 2)
    qc.measure(3, 3)

    shots = 1000
    job = simulate(qc, shots)

    counts = job.result().get_counts()

    debug(qc, counts, probs)
    assert len(probs) == w_state_size
    offset = 0.02
    upper_bound = (shots / w_state_size / shots) + offset
    lower_bound = (shots / w_state_size / shots) - offset
    for key, value in probs.items():
        assert value < upper_bound
        assert value > lower_bound


if __name__ == '__main__':
    qc_size = 4
    testWState(qc_size)
