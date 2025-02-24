from quantum_launcher.base.qasm_translator import QiskitTranslation, CirqTranslation, Translation
from qiskit import QuantumCircuit
from cirq import Circuit


def test_qiskit_to_cirq_translation():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.x(0)
    # translator_to = QiskitTranslation()
    # translator_from = CirqTranslation()
    # qasm = translator_to.to_qasm(circuit)
    # cirq_circuit = translator_from.from_qasm(qasm)
    cirq_circuit = Translation.get_translation(circuit, 'cirq')
    assert isinstance(cirq_circuit, Circuit)
