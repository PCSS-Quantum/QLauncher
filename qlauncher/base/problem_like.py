from collections.abc import Callable
from typing import Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.mappers import ParityMapper, QubitMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem


class ProblemLike: ...


class QUBO(ProblemLike):
	def __init__(self, matrix: np.ndarray, offset: float = 0) -> None:
		self.matrix = matrix
		self.offset = offset

	def to_hamiltonian(self) -> 'Hamiltonian':
		num_vars = self.matrix.shape[0]
		pauli = 0
		for i, col in enumerate(self.matrix):
			for j, entry in enumerate(col):
				if entry == 0:
					continue
				if i == j:
					pauli += SparsePauliOp.from_sparse_list([('I', [0], 0.5), ('Z', [i], -0.5)], num_vars) * entry
				else:
					pauli += (
						SparsePauliOp.from_sparse_list(
							[('I', [0], 0.25), ('Z', [i], -0.25), ('Z', [j], -0.25), ('ZZ', [i, j], 0.25)], num_vars
						)
						* entry
					)
		pauli += SparsePauliOp.from_sparse_list([('I', [], self.offset)], num_vars)
		return Hamiltonian(pauli)

	def to_fn(self) -> 'FN':
		def function(bin_vec: np.ndarray) -> float:
			return np.dot(bin_vec, np.dot(self.matrix, bin_vec)) + self.offset

		return FN(function)


class FN(ProblemLike):
	def __init__(self, function: Callable[[np.ndarray], float]) -> None:
		self.function = function

	def __call__(self, vector: np.ndarray) -> float:
		return self.function(vector)


class Hamiltonian(ProblemLike):
	def __init__(self, hamiltonian: SparsePauliOp) -> None:
		self.hamiltonian = hamiltonian
		self._mixer_hamiltonian: SparsePauliOp | None = None

	@property
	def mixer_hamiltonian(self) -> SparsePauliOp | None:
		return self._mixer_hamiltonian

	@mixer_hamiltonian.setter
	def mixer_hamiltonian(self, mixer_hamiltonian: SparsePauliOp) -> None:
		self._mixer_hamiltonian = mixer_hamiltonian

	def get_mixer_hamiltonian(self) -> None: ...

	def get_QAOAAnsatz_initial_state(self) -> None: ...


class BQM(ProblemLike):
	def __init__(self, bqm: Any) -> None:  # noqa: ANN401
		self.bqm = bqm


class Molecule(ProblemLike):
	def __init__(self, instance: MoleculeInfo, mapper: QubitMapper | None = None, basis_set: str = 'STO-6G') -> None:
		self.instance = instance
		self.basis_set = basis_set
		self.mapper = ParityMapper() if mapper is None else mapper
		self.problem: ElectronicStructureProblem = PySCFDriver.from_molecule(instance, basis=self.basis_set).run()
		self.mapper.num_particles = self.problem.num_particles
		operator = self.mapper.map(self.problem.hamiltonian.second_q_op())
		if not isinstance(operator, SparsePauliOp):
			raise TypeError
		self.operator: SparsePauliOp = operator

	@staticmethod
	def from_preset(instance_name: str) -> 'Molecule':
		match instance_name:
			case 'H2':
				instance = MoleculeInfo(['H', 'H'], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.74)])
			case _:
				raise ValueError(f"Molecule {instance_name} not supported, currently you can use: 'H2'")
		return Molecule(instance)
