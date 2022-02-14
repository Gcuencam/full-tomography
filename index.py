import math as m
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector

def main(): 
    # Use Aer's qasm_simulator
    simulator = QasmSimulator()

    # Create a Quantum Circuit acting on the q register
    circuit = QuantumCircuit(3,3)

    circuit.ry(2*np.arccos(1/m.sqrt(3)), 0)
    circuit.ch(0, 1)
    circuit.cx(1, 2)
    circuit.cx(0, 1)
    circuit.x(0)

    psi = Statevector.from_instruction(circuit)
    # Probabilities for measuring both qubits
    probs = psi.probabilities_dict()
    print('probs: {}'.format(probs))


    # compile the circuit down to low-level QASM instructions
    # supported by the backend (not needed for simple circuits)
    compiled_circuit = transpile(circuit, simulator)

    # Execute the circuit on the qasm simulator
    job = simulator.run(compiled_circuit, shots=100)

    # Grab results from the job
    result = job.result()

    # Returns counts
    # counts = result.get_counts(compiled_circuit)
    # print("\nResult:",counts)

    # Draw the circuit
    circuit.draw()

if __name__ == '__main__':
    main()