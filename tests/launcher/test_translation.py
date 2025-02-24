from quantum_launcher.base.qasm_translator import QiskitTranslation, CirqTranslation, Translation
import qiskit
import cirq


def test_qiskit_to_cirq_translation():
    circuit = qiskit.QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.x(0)
    translator_to = QiskitTranslation()
    translator_from = CirqTranslation()
    qasm = translator_to.to_qasm(circuit)
    cirq_circuit = translator_from.from_qasm(qasm)
    assert isinstance(cirq_circuit, cirq.Circuit)


def test_auto_translation():
    circuit = qiskit.QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.x(0)
    cirq_circuit = Translation.get_translation(circuit, 'cirq')
    assert isinstance(cirq_circuit, cirq.Circuit)
