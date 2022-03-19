import math

from qiskit import QuantumCircuit
from qiskit.circuit.library import Diagonal, QFT
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info.operators import Operator
from src.lib.quantum_commons import simulate
from src.lib.w_state import w_state


def get_m():
    alpha = math.sqrt((3 + math.sqrt(3)) / 6)
    beta = math.sqrt((3 - math.sqrt(3)) / 6)
    # a = m.sqrt(2) * alpha
    a = alpha
    # b = m.sqrt(2) * beta
    b = beta

    op = Operator([
        [a, b],
        [b, -a]
    ])
    return UnitaryGate(op, '~M')


def measure_qubit(qc, qubit, ancilla):
    qc.cx(ancilla, qubit)
    qc = qc.compose(get_m(), qubits=[qubit])
    qc = qc.compose(Diagonal([1, 1, 1, 1j]), qubits=[qubit, ancilla])
    qc = qc.compose(QFT(1), qubits=[ancilla])
    qc.measure(qubit, qubit)
    qc.measure(ancilla, ancilla)

    return qc


def measure_povm(qc):
    size = qc.num_qubits

    expanded_circuit = QuantumCircuit(size * 2, size * 2)
    expanded_circuit = expanded_circuit.compose(qc, [0, 1])

    expanded_circuit.barrier([0, 1, 2, 3])

    for i in range(qc.num_qubits):
        expanded_circuit = measure_qubit(expanded_circuit, i, i + size)

    return expanded_circuit


if __name__ == '__main__':
    size = 2
    w_qc = w_state(size)
    measured_circuit = measure_povm(w_qc)
    print(measured_circuit)

    job = simulate(measured_circuit, 1000)
    counts = job.result().get_counts()
    print(counts)
