import cirq
import qiskit

from qlauncher.routines.qiskit.backends.gate_circuit_backend import GateCircuitBackend


def test_auto_translation() -> None:
	circuit = qiskit.QuantumCircuit(2)
	circuit.h(0)
	circuit.cx(0, 1)
	circuit.x(0)
	cirq_circuit = GateCircuitBackend.get_translation(circuit, 'cirq')
	assert isinstance(cirq_circuit, cirq.Circuit)
