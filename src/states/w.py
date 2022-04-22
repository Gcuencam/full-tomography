# -*- coding: utf-8 -*-
import math as m

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from src.common.quantum_commons import simulate, debug_circuit


def build(circuit, referencePosition: int, n: int):
    if (n < 2):
        raise Exception('The size must be at least two.')
    if (referencePosition < 0):
        raise Exception('The reference position must be at least zero.')
    if (n > circuit.width() / 2):
        raise Exception('The circuit does not have enough qubits.')
    if (referencePosition + n > circuit.width() / 2):
        raise Exception('The reference position must be lower.')
    firstPosition = referencePosition
    lastPosition = referencePosition + n - 1
    _circuit = circuit.copy()
    for i in range(firstPosition, lastPosition + 1):
        _circuit.reset(i)
    if (n >= 3):
        _circuit.ry(2 * np.arccos(1 / m.sqrt(n)), firstPosition)
        for i in range(n - 3):
            _circuit.cry(2 * np.arccos(1 / m.sqrt(n - (i + 1))), firstPosition + i, firstPosition + i + 1)
    if (n == 2):
        _circuit.h(firstPosition)
    else:
        _circuit.ch(lastPosition - 2, lastPosition - 1)
    for i in reversed(range(firstPosition, lastPosition)):
        _circuit.cx(i, i + 1)
    _circuit.x(firstPosition)

    return _circuit


def testWState(qc_size, w_state_size):
    qc = QuantumCircuit(qc_size, qc_size)
    qc = build(qc, 0, w_state_size)

    psi = Statevector.from_instruction(qc)
    probs = psi.probabilities_dict()

    for i in range(w_state_size):
        qc.measure(i, i)

    shots = 1000
    job = simulate(qc, shots)

    counts = job.result().get_counts()

    debug_circuit(qc, counts, probs)
    assert len(probs) == w_state_size
    offset = 0.02
    upper_bound = (shots / w_state_size / shots) + offset
    lower_bound = (shots / w_state_size / shots) - offset
    for key, value in probs.items():
        assert value < upper_bound
        assert value > lower_bound


if __name__ == '__main__':
    qc_size = 3
    w_state_size = 3
    testWState(qc_size, w_state_size)
