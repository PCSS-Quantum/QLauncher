"""
Routine file for Cirq library
"""

import math
from collections.abc import Iterable, Sequence
from typing import Any, Literal

import numpy as np
import qiskit
from qiskit.primitives import BaseSamplerV2, BitArray, DataBin, Sampler
from qiskit.primitives.base.base_primitive_job import BasePrimitiveJob
from qiskit.primitives.base.sampler_result import SamplerResult
from qiskit.primitives.containers.primitive_result import PrimitiveResult
from qiskit.primitives.containers.sampler_pub import SamplerPubLike
from qiskit.primitives.containers.sampler_pub_result import SamplerPubResult
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.quantum_info import SparsePauliOp
from qiskit.result import QuasiDistribution

from qlauncher.base import Backend
from qlauncher.base.base import GateCircuitBackend
from qlauncher.exceptions import DependencyError
from qlauncher.routines.qiskit.mitigation_suppression.base import CircuitExecutionMethod
from qlauncher.routines.qiskit.mitigation_suppression.mitigation import NoMitigation

try:
	import cirq
	from cirq.contrib.qasm_import.qasm import circuit_from_qasm
	from cirq.sim.sparse_simulator import Simulator
except ImportError as e:
	raise DependencyError(e, install_hint='cirq') from e


def extract_bitstrings_from_result(result: cirq.Result) -> list[str]:
	measurements = result.measurements

	sorted_keys = list(measurements.keys())

	bitstrings = []
	num_shots = len(measurements[sorted_keys[0]])

	for shot_index in range(num_shots):
		bits = []
		for key in sorted_keys:
			bits.extend(str(b) for b in measurements[key][shot_index])

		bitstring = ''.join(bits)
		bitstrings.append(bitstring)
	return bitstrings


def cirq_result_to_counts(result: cirq.Result) -> dict:
	bitstrings = extract_bitstrings_from_result(result)

	counts = {}
	for bs in bitstrings:
		counts[bs] = counts.get(bs, 0) + 1
	return counts


def cirq_result_to_probabilities(result: cirq.Result, integer_keys: bool = False) -> dict:
	counts = cirq_result_to_counts(result)

	total_shots = sum(counts.values())
	return {int(k, 2): v / total_shots for k, v in counts.items()} if integer_keys else {k: v / total_shots for k, v in counts.items()}


class _CirqRunner:
	simulator = Simulator()
	repetitions = 1024

	@classmethod
	def calculate_circuit(
		cls, circuit: qiskit.QuantumCircuit, return_type: Literal['counts', 'dist', 'raw'] = 'counts', shots: int | None = None
	) -> dict | list[str]:
		if circuit.num_clbits == 0:
			circuit = circuit.measure_all(inplace=False)

		cirq_circ = GateCircuitBackend.get_translation(circuit, 'cirq')

		result = cls.simulator.run(cirq_circ, repetitions=cls.repetitions if shots is None else shots)

		if return_type == 'raw':
			return extract_bitstrings_from_result(result)

		return cirq_result_to_counts(result) if return_type == 'counts' else cirq_result_to_probabilities(result)


class CirqSampler(Sampler):
	"""Sampler adapter for Cirq"""

	def _call(self, circuits: Sequence[int], parameter_values: Sequence[Sequence[float]], **run_options) -> SamplerResult:
		bound_circuits = []
		for i, value in zip(circuits, parameter_values):
			bound_circuits.append(
				self._circuits[i] if len(value) == 0 else self._circuits[i].assign_parameters(dict(zip(self._parameters[i], value)))
			)
		distributions = [_CirqRunner.calculate_circuit(circuit, 'dist') for circuit in bound_circuits]
		quasi_dists = list(map(QuasiDistribution, distributions))
		return SamplerResult(quasi_dists, [{} for _ in range(len(parameter_values))])


class CirqSamplerV2(BaseSamplerV2):
	def __init__(self) -> None:
		super().__init__()
		self.cirq_sampler_v1 = CirqSampler()

	def _run(self, bound_circuits, shots):
		bitstring_collections = [_CirqRunner.calculate_circuit(circuit, 'raw', shots) for circuit in bound_circuits]

		results = []
		for bsc in bitstring_collections:
			arr = np.array([np.frombuffer(int(bs, 2).to_bytes(math.ceil(len(bs) / 8)), dtype=np.uint8) for bs in bsc])
			bit_array = BitArray(arr, num_bits=len(bsc[0]))
			results.append(SamplerPubResult(data=DataBin(meas=bit_array), metadata={'shots': len(bitstring_collections[0])}))
		return PrimitiveResult(results, metadata={'version': 2})

	def run(self, pubs: Iterable[SamplerPubLike], *, shots: int | None = None) -> BasePrimitiveJob[PrimitiveResult[SamplerPubResult], Any]:
		bound_circuits = []
		for pub in pubs:
			if isinstance(pub, qiskit.QuantumCircuit):
				bound_circuits.append(pub)
			elif len(pub) == 1 and isinstance(pub[0], qiskit.QuantumCircuit):
				bound_circuits.append(pub[0])
			elif len(pub) == 2:
				bound_circuits.append(pub[0].assign_parameters(pub[1]))
			else:
				raise ValueError(
					f'Incorrect pub, expected QuantumCircuit, tuple[QuantumCircuit] or tuple[QuantumCircuit, Iterable[float]], got {type(pub)}'
				)

		job = PrimitiveJob(self._run, bound_circuits, shots)
		job._submit()
		return job


class CirqBackend(GateCircuitBackend):
	"""

	Args:
		Backend (_type_): _description_
	"""

	basis_gates = ['x', 'y', 'z', 'cx', 'h', 'rx', 'ry', 'rz']
	compatible_circuit = cirq.Circuit
	language = 'cirq'

	def __init__(
		self,
		name: Literal['local_simulator'] = 'local_simulator',
		error_mitigation_strategy: CircuitExecutionMethod | None = None,
	):
		self.sampler = CirqSamplerV2()
		self.samplerV1 = CirqSampler()
		self._mitigation_strategy = error_mitigation_strategy if error_mitigation_strategy is not None else NoMitigation()
		self.backendv1v2 = None
		super().__init__(name)

	@staticmethod
	def to_qasm(circuit: cirq.Circuit) -> str:
		return circuit.to_qasm()

	@staticmethod
	def from_qasm(qasm: str) -> cirq.Circuit:
		return circuit_from_qasm(qasm)

	def sample_circuit(self, circuit: qiskit.QuantumCircuit | cirq.Circuit, shots: int = 1024) -> dict[str, int]:
		return self._mitigation_strategy.sample(circuit, self, shots)

	def estimate_energy(self, circuit: qiskit.QuantumCircuit, observable: SparsePauliOp) -> float:
		raise NotImplementedError
