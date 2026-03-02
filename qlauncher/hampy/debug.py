"""Module with functionalities for debugging Hamiltonians and checking their boolean properties"""

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import SparsePauliOp

from qlauncher.hampy.object import Equation


class TruthTable:
	"""
	Generates and analyzes a full truth table for a Hamiltonian represented as
	either an :class:`Equation` or a :class:`qiskit.quantum_info.SparsePauliOp`.

	The class evaluates the Hamiltonian on all possible bitstrings of length
	``size`` and exposes utilities for debugging, visualizing value
	distributions, and checking boolean properties (e.g., whether the
	Hamiltonian is binary-valued).

	Parameters
	----------
	equation : Equation or SparsePauliOp
		Hamiltonian to evaluate. If an :class:`Equation` is provided, its
		simplified SparsePauliOp is used. If a SparsePauliOp is provided,
		the number of qubits is inferred.
	return_int : bool, optional
		Whether to cast diagonal values to integers. Defaults to ``True``.

	Examples
	--------
	From an Equation:
	>>> eq = Equation(2)
	>>> x0, x1 = eq[0], eq[1]
	>>> tt = TruthTable(x0 ^ x1)
	>>> tt.truth_table
	{'00': 0, '01': 1, '10': 1, '11': 0}

	From a SparsePauliOp:
	>>> from qiskit.quantum_info import SparsePauliOp
	>>> H = SparsePauliOp.from_sparse_list([('Z', [0], 1.0)], 1)
	>>> tt = TruthTable(H)
	>>> tt['0'], tt['1']
	(1, -1)

	Get solutions for a given value:
	>>> tt.get_solutions(tt.lowest_value)
	['1']

	Check if the Hamiltonian is binary-valued:
	>>> tt.check_if_binary()  # Hamiltonian is binary if all energy values ar either 0 or 1
	False

	Plot the distribution of output energies:
	>>> tt.plot_distribution()

	Access row by integer or bitstring:
	>>> tt[0]
	1
	>>> tt['1']
	-1
	"""

	def __init__(self, equation: Equation | SparsePauliOp, return_int: bool = True):
		if isinstance(equation, SparsePauliOp):
			hamiltonian = equation
			size = hamiltonian.num_qubits
			if not isinstance(size, int):
				raise TypeError('Cannot read number of qubits from provided SparsePauliOp')
		elif isinstance(equation, Equation):
			hamiltonian = equation.hamiltonian
			size = equation.size
		self.size = size
		self.return_int = return_int
		self.truth_table = self._ham_to_truth(hamiltonian)
		self.lowest_value = min(self.truth_table.values())

	def count(self, value: int) -> int:
		return list(self.truth_table.values()).count(value)

	def get_solutions(self, value: int) -> list[str]:
		return list(filter(lambda x: self.truth_table[x] == value, self.truth_table.keys()))

	def count_min_value_solutions(self) -> int:
		return self.count(self.lowest_value)

	def get_min_value_solutions(self) -> list[str]:
		return self.get_solutions(self.lowest_value)

	def check_if_binary(self) -> bool:
		return all((value == 0 or value == 1) for value in self.truth_table.values())

	def plot_distribution(self) -> None:
		values = list(self.truth_table.values())
		counts, bins = np.histogram(values, max(values) + 1)
		plt.stairs(counts, bins)
		plt.show()

	def _ham_to_truth(self, hamiltonian: SparsePauliOp) -> dict[str, int]:
		return {
			''.join(reversed(bitstring)): value
			for bitstring, value in zip(
				product(('0', '1'), repeat=self.size),
				(int(x.real) for x in hamiltonian.to_matrix().diagonal()) if self.return_int else hamiltonian.to_matrix().diagonal(),
				strict=True,
			)
		}

	def __getitem__(self, index: str | int):
		if isinstance(index, int):
			index = bin(index)[2:].zfill(self.size)
		return self.truth_table[index]
