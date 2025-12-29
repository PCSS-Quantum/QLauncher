import pytest
import qiskit

from qlauncher.exceptions import DependencyError
from qlauncher.routines.qiskit import QiskitBackend
from qlauncher.routines.qiskit.backends.gate_circuit_backend import GateCircuitBackend

try:
	import cirq

	from qlauncher.routines.cirq import CirqBackend

	CIRQ = True
except (DependencyError, ImportError):
	CIRQ = False


@pytest.mark.skipif(not CIRQ, reason='cirq not installed')
def test_auto_translation() -> None:
	circuit = qiskit.QuantumCircuit(2)
	circuit.h(0)
	circuit.cx(0, 1)
	circuit.x(0)
	cirq_circuit = GateCircuitBackend.get_translation(circuit, cirq.Circuit)
	assert isinstance(cirq_circuit, cirq.Circuit)


@pytest.mark.skipif(not CIRQ, reason='cirq not installed')
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
