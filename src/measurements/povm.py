import math

from qiskit import QuantumCircuit
from qiskit.circuit.library import Diagonal
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info.operators import Operator

from src.measurement_collector import build
from src.common.quantum_commons import simulate
from src.states.builder import States


def get_m():
    dividing = 12
    alpha = math.sqrt((3 + math.sqrt(3)) / dividing)
    beta = math.sqrt((3 - math.sqrt(3)) / dividing)
    a = math.sqrt(2) * alpha
    b = math.sqrt(2) * beta

    op = Operator([
        [a, b],
        [b, -a]
    ])
    return UnitaryGate(op, '~M')


def measure_qubit(qc, qubit, ancilla):
    qc.cx(qubit, ancilla)
    qc = qc.compose(get_m(), qubits=[ancilla])
    qc = qc.compose(Diagonal([1, 1, 1, 1j]), qubits=[ancilla, qubit])
    # qc = qc.compose(CPhaseGate(numpy.pi / 2), qubits=[ancilla, qubit])
    # qc = qc.compose(QFT(1), qubits=[qubit])
    qc.h(qubit)
    qc.measure(qubit, 2 * qubit)
    qc.measure(ancilla, (2 * qubit) + 1)

    return qc


def measure_povm(qc):
    size = qc.num_qubits

    expanded_circuit = QuantumCircuit(size * 2, size * 2)
    expanded_circuit = expanded_circuit.compose(qc, range(0, size))

    expanded_circuit.barrier(range(0, size))

    for i in range(qc.num_qubits):
        expanded_circuit = measure_qubit(expanded_circuit, i, i + size)

    return expanded_circuit


if __name__ == '__main__':
    qc_size = 2
    w_qc = build(States.W, QuantumCircuit(qc_size, qc_size), 0, qc_size)
    measured_circuit = measure_povm(w_qc)
    print(measured_circuit)

    job = simulate(measured_circuit, 1000)
    counts = job.result().get_counts()
    print(counts)
