from collections.abc import Callable
from typing import Any

import qiskit
from qiskit.primitives import BaseSamplerV2
from qiskit.transpiler.passes import RemoveBarriers

from qlauncher.base import Backend


class GateCircuitBackend(Backend):
	sampler: BaseSamplerV2

	basis_gates: list[str] = []
	language: str = 'None'
	compatible_circuit: type
	language_name_mapping: dict[str, 'GateCircuitBackend'] = {}
	circuit_class_mapping: dict[type, 'GateCircuitBackend'] = {}
	language_circuit_mapping: dict[str, type] = {}
	circuit_language_mapping: dict[type, str] = {}

	to_qasm: Callable
	from_qasm: Callable

	def __init_subclass__(cls):
		GateCircuitBackend.language_name_mapping[cls.language] = cls('local_simulator')
		GateCircuitBackend.circuit_class_mapping[cls.compatible_circuit] = cls('local_simulator')
		GateCircuitBackend.language_circuit_mapping[cls.language] = cls.compatible_circuit
		GateCircuitBackend.circuit_language_mapping[cls.compatible_circuit] = cls.language

	@staticmethod
	def get_translation(circuit: Any, language: str) -> Any:
		"""Transpiles circuit into given languages basis_gates, translates it to qasm, and from qasm into desired languages object."""
		print(circuit.__class__)
		circuit_qasm_translator = GateCircuitBackend.circuit_class_mapping[circuit.__class__]
		qasm_circuit_translator = GateCircuitBackend.language_name_mapping[language]
		if isinstance(circuit, qiskit.QuantumCircuit):
			transpiled_circuit = GateCircuitBackend.transpile_circuit(circuit, qasm_circuit_translator.basis_gates)
		else:
			transpiled_circuit = circuit  # Transpilation is usually not needed
		qasm = circuit_qasm_translator.to_qasm(transpiled_circuit)
		return qasm_circuit_translator.from_qasm(qasm)

	@staticmethod
	def transpile_circuit(qc: qiskit.QuantumCircuit, basis_gates: list[str]) -> qiskit.QuantumCircuit:
		"""Makes circuit compatible with cirq

		Args:
			qc (qiskit.QuantumCircuit): circuit

		Returns:
			qiskit.QuantumCircuit: transpiled circuit
		"""
		qc_transpiled = qiskit.transpile(qc, basis_gates=basis_gates, optimization_level=3)

		return RemoveBarriers()(qc_transpiled)
