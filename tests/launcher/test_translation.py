import cirq
import qiskit

from qlauncher.routines.cirq import CirqBackend
from qlauncher.routines.qiskit import QiskitBackend
from qlauncher.routines.qiskit.backends.gate_circuit_backend import GateCircuitBackend


def test_auto_translation() -> None:
	circuit = qiskit.QuantumCircuit(2)
	circuit.h(0)
	circuit.cx(0, 1)
	circuit.x(0)
	cirq_circuit = GateCircuitBackend.get_translation(circuit, cirq.Circuit)
	assert isinstance(cirq_circuit, cirq.Circuit)


def test_translating_samplers() -> None:
	qiskit_circuit = qiskit.QuantumCircuit(2)
	qiskit_circuit.h(0)
	qiskit_circuit.cx(0, 1)
	qiskit_circuit.x(0)
	qiskit_circuit.measure_all()
	cirq_circuit = GateCircuitBackend.get_translation(qiskit_circuit, cirq.Circuit)
	qiskit_backend = QiskitBackend()
	cirq_backend = CirqBackend()
	assert isinstance(cirq_backend.sample_circuit(qiskit_circuit), dict)
	assert isinstance(qiskit_backend.sample_circuit(cirq_circuit), dict)
	assert isinstance(qiskit_backend.sample_circuit(qiskit_circuit), dict)
	assert isinstance(cirq_backend.sample_circuit(cirq_circuit), dict)
