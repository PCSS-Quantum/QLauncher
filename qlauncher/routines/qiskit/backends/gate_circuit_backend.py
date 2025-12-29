from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, get_args, get_origin

import qiskit
from qiskit.primitives import BaseSamplerV1, BaseSamplerV2
from qiskit.transpiler.passes import RemoveBarriers

from qlauncher.base import Backend
from qlauncher.routines.circuits import CIRCUIT_FORMATS

_AllowedCircuit = TypeVar('_AllowedCircuit', bound=CIRCUIT_FORMATS)


class GateCircuitBackend(Backend, Generic[_AllowedCircuit], ABC):
	sampler: BaseSamplerV2
	samplerV1: BaseSamplerV1

	basis_gates: list[str] = []
	circuit_class_mapping: dict[type, 'GateCircuitBackend'] = {}

	def __init_subclass__(cls):
		super().__init_subclass__()

		if GateCircuitBackend not in cls.__bases__:  # Skip subclasses of subclasses and further
			return

		for base in getattr(cls, '__orig_bases__', ()):
			origin = get_origin(base)
			if origin is GateCircuitBackend:
				(allowed_circuit,) = get_args(base)
				cls.compatible_circuit = allowed_circuit
				GateCircuitBackend.circuit_class_mapping[cls.compatible_circuit] = cls('local_simulator')

	@staticmethod
	def get_translation(circuit: CIRCUIT_FORMATS, output_format: type[CIRCUIT_FORMATS]) -> CIRCUIT_FORMATS:
		"""Transpiles circuit into given languages basis_gates, translates it to qasm, and from qasm into desired languages object."""
		circuit_qasm_translator = GateCircuitBackend.circuit_class_mapping[circuit.__class__]
		qasm_circuit_translator = GateCircuitBackend.circuit_class_mapping[output_format]

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

	@staticmethod
	@abstractmethod
	def to_qasm(circuit: _AllowedCircuit) -> str:
		pass

	@staticmethod
	@abstractmethod
	def from_qasm(qasm: str) -> _AllowedCircuit:
		pass

	@abstractmethod
	def sample_circuit(self, circuit: CIRCUIT_FORMATS, shots: int = 1024) -> dict[str, int]:
		pass
