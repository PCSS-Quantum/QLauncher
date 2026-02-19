from __future__ import annotations

from collections import defaultdict
from itertools import chain
from typing import TYPE_CHECKING, Literal

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit._accelerate.circuit import CircuitInstruction as AccelerateInstruction
from qiskit.circuit import Instruction, Operation

from qlauncher.routines.circuits import CIRCUIT_FORMATS
from qlauncher.utils import sum_counts

from .base import CircuitExecutionMethod

if TYPE_CHECKING:
	from qiskit.quantum_info import SparsePauliOp

	from qlauncher.routines.qiskit.adapters import GateCircuitBackend
	from qlauncher.routines.qiskit.backends.qiskit_backend import QiskitBackend


class NoMitigation(CircuitExecutionMethod):
	compatible_circuit = CIRCUIT_FORMATS

	def sample(self, circuit: CIRCUIT_FORMATS, backend: GateCircuitBackend, shots: int = 1024) -> dict[str, int]:
		return backend.sampler.run([circuit], shots=shots).result()[0].join_data().get_counts()

	def estimate(self, circuit: QuantumCircuit, observable: SparsePauliOp, backend: QiskitBackend) -> float:
		return backend.estimator.run([(circuit, observable)]).result()[0].data.evs


class WeighedMitigation(CircuitExecutionMethod):
	def __init__(self, mitigation_methods: list[CircuitExecutionMethod], method_weights: list[float] | None = None) -> None:
		if method_weights is None:
			method_weights = [1.0] * len(mitigation_methods)
		if len(method_weights) != len(mitigation_methods):
			raise ValueError(
				f'You must provide as many weights as there are methods! Expected {len(mitigation_methods)}, got {len(method_weights)}'
			)
		self.weights = method_weights
		self.methods = mitigation_methods
		super().__init__()

	def sample(self, circuit: QuantumCircuit, backend: QiskitBackend, shots: int = 1024) -> dict[str, int]:
		counts = (m.sample(circuit, backend, shots) for m in self.methods)
		result = defaultdict(int)
		for count, weight in zip(counts, self.weights, strict=True):
			for k, v in count.items():
				result[k] += int(round(v * weight, 0))
		return result

	def estimate(self, circuit: QuantumCircuit, observable: SparsePauliOp, backend: QiskitBackend) -> float:
		return sum(m.estimate(circuit, observable, backend) * w for m, w in zip(self.methods, self.weights, strict=True)) / len(
			self.weights
		)


class PauliTwirling(CircuitExecutionMethod):
	"""
	Error mitigation technique based on averaging the results of running multiple "twirled" versions of the initial circuit.
	The method appends additional gates on both sides of random 2 qubit gates (cx, ecr).
	"""

	compatible_circuit = QuantumCircuit

	def __init__(self, num_random_circuits: int, max_substitute_gates_per_circuit: int = 4, do_transpile: bool = True) -> None:
		self.num_random_circuits = num_random_circuits
		self.max_substitute_gates_per_circuit = max_substitute_gates_per_circuit
		self.do_transpile = do_transpile

	def _random_replacement_op(self, inst: AccelerateInstruction) -> list[AccelerateInstruction]:
		op: Operation = inst.operation
		match op.name:
			case 'cx':
				return [
					[
						AccelerateInstruction(
							operation=Instruction(name='x', num_qubits=1, num_clbits=0, params=[]), qubits=[inst.qubits[0]]
						),
						AccelerateInstruction(
							operation=Instruction(name='y', num_qubits=1, num_clbits=0, params=[]), qubits=[inst.qubits[1]]
						),
						inst,
						AccelerateInstruction(
							operation=Instruction(name='y', num_qubits=1, num_clbits=0, params=[]), qubits=[inst.qubits[0]]
						),
						AccelerateInstruction(
							operation=Instruction(name='z', num_qubits=1, num_clbits=0, params=[]), qubits=[inst.qubits[1]]
						),
					],
					[
						AccelerateInstruction(
							operation=Instruction(name='x', num_qubits=1, num_clbits=0, params=[]), qubits=[inst.qubits[0]]
						),
						inst,
						AccelerateInstruction(
							operation=Instruction(name='x', num_qubits=1, num_clbits=0, params=[]), qubits=[inst.qubits[0]]
						),
						AccelerateInstruction(
							operation=Instruction(name='x', num_qubits=1, num_clbits=0, params=[]), qubits=[inst.qubits[1]]
						),
					],
					[
						AccelerateInstruction(
							operation=Instruction(name='z', num_qubits=1, num_clbits=0, params=[]), qubits=[inst.qubits[0]]
						),
						inst,
						AccelerateInstruction(
							operation=Instruction(name='z', num_qubits=1, num_clbits=0, params=[]), qubits=[inst.qubits[0]]
						),
					],
				][int(np.random.default_rng().integers(0, 3))]

			case 'ecr':
				return [
					[
						AccelerateInstruction(
							operation=Instruction(name='x', num_qubits=1, num_clbits=0, params=[]), qubits=[inst.qubits[0]]
						),
						AccelerateInstruction(
							operation=Instruction(name='y', num_qubits=1, num_clbits=0, params=[]), qubits=[inst.qubits[1]]
						),
						inst,
						AccelerateInstruction(
							operation=Instruction(name='x', num_qubits=1, num_clbits=0, params=[]), qubits=[inst.qubits[0]]
						),
						AccelerateInstruction(
							operation=Instruction(name='y', num_qubits=1, num_clbits=0, params=[]), qubits=[inst.qubits[1]]
						),
					],
					[
						AccelerateInstruction(
							operation=Instruction(name='x', num_qubits=1, num_clbits=0, params=[]), qubits=[inst.qubits[0]]
						),
						AccelerateInstruction(
							operation=Instruction(name='z', num_qubits=1, num_clbits=0, params=[]), qubits=[inst.qubits[1]]
						),
						inst,
						AccelerateInstruction(
							operation=Instruction(name='x', num_qubits=1, num_clbits=0, params=[]), qubits=[inst.qubits[0]]
						),
						AccelerateInstruction(
							operation=Instruction(name='z', num_qubits=1, num_clbits=0, params=[]), qubits=[inst.qubits[1]]
						),
					],
					[
						AccelerateInstruction(
							operation=Instruction(name='x', num_qubits=1, num_clbits=0, params=[]), qubits=[inst.qubits[1]]
						),
						inst,
						AccelerateInstruction(
							operation=Instruction(name='x', num_qubits=1, num_clbits=0, params=[]), qubits=[inst.qubits[1]]
						),
					],
				][int(np.random.default_rng().integers(0, 3))]
			case _:
				return [inst]

	def _twirl_circuit(self, transpiled_circuit: QuantumCircuit) -> QuantumCircuit:
		"""Apply random self.max_substitute_gates_per_circuit twirls on random (no replacement) gates of the circuit."""
		circuit = transpiled_circuit.copy()

		double_gates_with_indices: list[tuple[int, AccelerateInstruction]] = [
			(i, x) for i, x in enumerate(circuit.data) if x.operation.num_qubits == 2
		]

		choice_idxs = np.random.default_rng().choice(
			range(len(double_gates_with_indices)),
			size=min(self.max_substitute_gates_per_circuit, len(double_gates_with_indices)),
			replace=False,
		)

		data_cpy = [[x] for x in circuit.data]

		for i in choice_idxs:
			data_cpy[i] = self._random_replacement_op(data_cpy[i][0])

		circuit.data = list(chain.from_iterable(data_cpy))  # Collapse [[e1],[e2,e3],[e4],...] to [e1,e2,e3,e4,...]

		return circuit

	def _get_workable_circuit(self, circuit: QuantumCircuit, backend: QiskitBackend) -> QuantumCircuit:
		"""Get either transpiled circuit if do_transpile is set or copy of circuit you can change as you wish."""
		return transpile(circuit, basis_gates=list(backend.backendv1v2.target.operation_names)) if self.do_transpile else circuit.copy()

	def sample(self, circuit: QuantumCircuit, backend: QiskitBackend, shots: int = 1024) -> dict[str, int]:
		input_circ = self._get_workable_circuit(circuit, backend)
		results = backend.sampler.run(
			[transpile(self._twirl_circuit(input_circ), backend.backendv1v2) for _ in range(self.num_random_circuits)],
			shots=shots // self.num_random_circuits,
		).result()

		counts = [r.join_data().get_counts() for r in results]

		return sum_counts(*counts)

	def estimate(self, circuit: QuantumCircuit, observable: SparsePauliOp, backend: QiskitBackend) -> float:
		input_circ = self._get_workable_circuit(circuit, backend)

		results = backend.estimator.run(
			[
				(transpile(self._twirl_circuit(input_circ), basis_gates=list(backend.backendv1v2.target.operation_names)), observable)
				for _ in range(self.num_random_circuits)
			],
		).result()

		sum_evs = 0
		for r in results:
			sum_evs += r.data.evs

		return sum_evs / self.num_random_circuits


class ZeroNoiseExtrapolation(CircuitExecutionMethod):
	"""
	Error mitigation technique based on fitting a model to data generated
	by running a circuit made to multiply the error of the original circuit,
	then predicting the values at x=0 (original circuit)
	"""

	compatible_circuit = QuantumCircuit

	def __init__(self, num_extrapolations: int = 4, polynomial_degree: int = 3, mode: Literal['linear', 'exponential'] = 'linear') -> None:
		"""
		Args:
			num_extrapolations (int, optional): Number of times the whole circuit is repeated for the largest X. Defaults to 4.
			polynomial_degree (int, optional): Degree of fitted polynomial. Defaults to 3.
			mode (Literal[&quot;linear&quot;, &quot;exponential&quot;], optional):
				Scaling method. "linear" keeps the original values as is,
				"exponential" applies log before fitting model then applies exp to the model prediction.
				Defaults to "linear".

		Raises:
			ValueError: If the polynomial degree is larger or equal to the number of data points (num_extrapolations)
		"""
		super().__init__()
		if polynomial_degree >= num_extrapolations:
			raise ValueError('Degree must be lower than number of data points.')
		self.num_extrapolations = num_extrapolations
		self.degree = polynomial_degree
		self.mode: Literal['linear', 'exponential'] = mode

	def _get_repeated_circuits(self, circuit: QuantumCircuit) -> list[QuantumCircuit]:
		result = []
		meas_circ = circuit.copy()

		mod_circuit: QuantumCircuit = circuit.remove_final_measurements(inplace=False)
		inv = mod_circuit.inverse(annotated=True)

		for _ in range(1, self.num_extrapolations + 1):
			meas_circ.compose(inv, front=True, inplace=True)
			meas_circ.compose(mod_circuit, front=True, inplace=True)
			result.append(meas_circ.copy())

		return result

	def _get_zero_estimate(self, y_values: np.ndarray) -> float:
		return np.polynomial.Polynomial.fit(np.array(range(1, self.num_extrapolations + 1)), y_values, self.degree).convert().coef[0]

	def _get_zero_estimate_sampling(self, y_values: np.ndarray) -> np.ndarray:
		return np.array([self._get_zero_estimate(y_values.T[i]) for i in range(len(y_values[0]))])

	def _get_np_array_from_counts_dict(self, int_counts: dict[int, int], num_measured: int) -> np.ndarray:
		result = np.zeros(2**num_measured)
		for value, counts in int_counts.items():
			result[value] = counts
		return result

	def sample(self, circuit: QuantumCircuit, backend: GateCircuitBackend, shots: int = 1024) -> dict[str, int]:
		counts = np.array([
			self._get_np_array_from_counts_dict(res.join_data().get_int_counts(), circuit.num_clbits)
			for res in backend.sampler.run(self._get_repeated_circuits(circuit), shots=shots).result()
		])

		if self.mode == 'exponential':
			counts = np.log(counts)
			counts_fit = np.exp(self._get_zero_estimate_sampling(counts))
		else:
			counts_fit = np.maximum(self._get_zero_estimate_sampling(counts), 0) + (10 if counts.sum() == 0 else 0)

		counts_dict = {}
		for i, val in enumerate(counts_fit):
			counts_dict[np.binary_repr(i, circuit.num_clbits)] = int(val)

		return counts_dict

	def estimate(self, circuit: QuantumCircuit, observable: SparsePauliOp, backend: QiskitBackend) -> float:
		evs = np.array([
			res.data.evs for res in backend.estimator.run([(x, observable) for x in self._get_repeated_circuits(circuit)]).result()
		])
		if self.mode == 'exponential':
			evs = np.log(evs)
			return np.exp(self._get_zero_estimate(evs))
		return self._get_zero_estimate(evs)
