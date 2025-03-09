from quantum_launcher.base.translator import QiskitTranslation, CirqTranslation, Translation
import qiskit
import cirq


def test_qiskit_to_cirq_translation():
    circuit = qiskit.QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.x(0)
    translator_qiskit = QiskitTranslation()
    translator_cirq = CirqTranslation()
    qasm = translator_qiskit.to_qasm(circuit)
    cirq_circuit = translator_cirq.from_qasm(qasm)
    assert isinstance(cirq_circuit, cirq.Circuit)
    cirq_qasm = translator_cirq.to_qasm(cirq_circuit)
    assert isinstance(cirq_qasm, str)
    qiskit_circuit = translator_qiskit.from_qasm(cirq_qasm)
    assert isinstance(qiskit_circuit, qiskit.QuantumCircuit)
    assert circuit.decompose(reps=5) == qiskit_circuit.decompose(reps=5)


def test_auto_translation():
    circuit = qiskit.QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.x(0)
    cirq_circuit = Translation.get_translation(circuit, 'cirq')
    assert isinstance(cirq_circuit, cirq.Circuit)
