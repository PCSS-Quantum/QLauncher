from collections.abc import Callable

import numpy as np
from qiskit.quantum_info import SparsePauliOp


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
