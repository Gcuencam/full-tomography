import math as m
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector

from src.lib.wState import buildWState


def main():

    # Use Aer's qasm_simulator
    simulator = QasmSimulator()

    # Create a Quantum Circuit acting on the q register
    q = 5
    circuit = QuantumCircuit(q, q)
    circuit = buildWState(circuit, 0, 2)

    psi = Statevector.from_instruction(circuit)
    # Probabilities for measuring qubits
    probs = psi.probabilities_dict()
    print('probs: {}'.format(probs))

    # Compile the circuit down to low-level QASM instructions
    # supported by the backend
    compiled_circuit = transpile(circuit, simulator)

    # Execute the circuit on the qasm simulator
    job = simulator.run(compiled_circuit, shots=1000)

    # Grab results from the job
    results = job.result()

    # Draw the circuit
    circuit.draw()


if __name__ == '__main__':
    main()
