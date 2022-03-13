import math

from qiskit import QuantumCircuit
from qiskit.circuit.library import Diagonal, QFT
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info.operators import Operator

from src.lib.quantum_commons import simulate


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


def tetrahedron_povm():
    size = 2

    circuit = QuantumCircuit(size, size)
    circuit.cx(1, 0)
    circuit = circuit.compose(get_m(), qubits=[0])
    circuit = circuit.compose(Diagonal([1, 1, 1, 1j]), qubits=[0, 1])
    circuit = circuit.compose(QFT(1), qubits=[1])
    circuit.measure(0, 0)
    circuit.measure(1, 1)

    print(circuit)

    job = simulate(circuit, 1000)
    counts = job.result().get_counts()
    print(counts)


if __name__ == '__main__':
    tetrahedron_povm()
