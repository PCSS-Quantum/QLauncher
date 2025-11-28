import numpy as np
from qiskit.quantum_info import SparsePauliOp

from qlauncher.base.problem_like import QUBO, Hamiltonian


def get_hamiltonian() -> Hamiltonian:
	return Hamiltonian(SparsePauliOp.from_list([('ZZ', -1), ('ZI', 2), ('IZ', 2), ('II', -1)]))


def get_qubo() -> QUBO:
	return QUBO(np.array([[1, 0], [-10, 1]]), 2)
