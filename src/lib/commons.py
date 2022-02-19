from qiskit import transpile
from qiskit.providers.aer import QasmSimulator


def simulate(qc, shots):
    simulator = QasmSimulator()
    compiled_circuit = transpile(qc, simulator)
    return simulator.run(compiled_circuit, shots=shots)

def debug(qc, counts, probs):
    print(qc)
    print(counts)
    print('probs: {}'.format(probs))