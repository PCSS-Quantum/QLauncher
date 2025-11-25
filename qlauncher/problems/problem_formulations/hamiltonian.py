"""Hamiltonian formulation of problems"""

from itertools import product

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import from_ising

import qlauncher.hampy as hampy
import qlauncher.problems.problem_initialization as problems
from qlauncher.base import formatter
from qlauncher.base.adapter_structure import adapter
from qlauncher.hampy import Equation, Variable
from qlauncher.problems.problem_formulations.hamiltonians.tsp import problem_to_hamiltonian as tsp_to_hamiltonian


@adapter('hamiltonian', 'qubo', onehot='quadratic')
def hamiltonian_to_qubo(hamiltonian):
	qp = from_ising(hamiltonian)
	conv = QuadraticProgramToQubo()
	qubo = conv.convert(qp).objective
	return qubo.quadratic.to_array(), 0


# @adapter('qubo', 'hamiltonian')
# def qubo_to_hamiltonian(qubo: np.ndarray) -> SparsePauliOp:
# 	q_matrix, offset = qubo
# 	num_vars = q_matrix.shape[0]
# 	pauli = 0
# 	for i, col in enumerate(q_matrix):
# 		for j, entry in enumerate(col):
# 			if entry == 0:
# 				continue
# 			if i == j:
# 				pauli += SparsePauliOp.from_sparse_list([('I', [0], 0.5), ('Z', [i], -0.5)], num_vars) * entry
# 			else:
# 				pauli += (
# 					SparsePauliOp.from_sparse_list([('I', [0], 0.25), ('Z', [i], -0.25), ('Z', [j], -0.25), ('ZZ', [i, j], 0.25)], num_vars)
# 					* entry
# 				)
# 	pauli += SparsePauliOp.from_sparse_list([('I', [], offset)], num_vars)
# 	return pauli


def ring_ham(ring: set, n):
	total = None
	ring = list(ring)
	for index in range(len(ring) - 1):
		sparse_list = []
		sparse_list.append(('XX', [ring[index], ring[index + 1]], 1))
		sparse_list.append(('YY', [ring[index], ring[index + 1]], 1))
		sp = SparsePauliOp.from_sparse_list(sparse_list, n)
		if total is None:
			total = sp
		else:
			total += sp
	sparse_list = []
	sparse_list.append(('XX', [ring[-1], ring[0]], 1))
	sparse_list.append(('YY', [ring[-1], ring[0]], 1))
	sp = SparsePauliOp.from_sparse_list(sparse_list, n)
	total += sp
	return SparsePauliOp(total)


@formatter(problems.JSSP, 'hamiltonian')
def get_qiskit_hamiltonian(problem: problems.JSSP) -> SparsePauliOp:
	if problem.optimization_problem:
		return problem.h_o
	return problem.h_d


@formatter(problems.Raw, 'hamiltonian')
def get_qiskit_hamiltonian(problem: problems.Raw) -> SparsePauliOp:
	return problem.instance


# @formatter(problems.TSP, 'hamiltonian')
# def get_qiskit_hamiltonian(problem: problems.TSP, onehot='exact', constraints_weight=1, costs_weight=1) -> SparsePauliOp:
# 	return tsp_to_hamiltonian(problem, onehot=onehot, constraints_weight=constraints_weight, costs_weight=costs_weight)


# @formatter(problems.GraphColoring, 'hamiltonian')
# def get_qiskit_hamiltonian(problem: problems.GraphColoring, constraints_weight=1, costs_weight=1):
# 	color_bit_length = int(np.ceil(np.log2(problem.num_colors)))
# 	num_qubits = problem.instance.number_of_nodes() * color_bit_length
# 	eq = Equation(num_qubits)
# 	# Penalty for assigning the same colors to neighboring vertices
# 	for node1, node2 in problem.instance.edges:
# 		for ind, comb in enumerate(product(range(2), repeat=color_bit_length)):
# 			if ind >= problem.num_colors:
# 				break
# 			eq2 = None
# 			for i in range(color_bit_length):
# 				qubit1 = eq[node1 * color_bit_length + i]
# 				qubit2 = eq[node2 * color_bit_length + i]
# 				exp = qubit1 & qubit2 if comb[i] else ~qubit1 & ~qubit2
# 				if eq2 is None:
# 					eq2 = exp
# 				else:
# 					eq2 &= exp
# 			eq += eq2
# 	eq *= costs_weight
# 	# Penalty for using excessive colors
# 	for node in problem.instance.nodes:
# 		for ind, comb in enumerate(product(range(2), repeat=color_bit_length)):
# 			if ind < problem.num_colors:
# 				continue
# 			eq2 = None
# 			for i in range(color_bit_length):
# 				qubit = eq[node * color_bit_length + i]
# 				exp = qubit if comb[i] else ~qubit
# 				if eq2 is None:
# 					eq2 = exp
# 				else:
# 					eq2 &= exp
# 			eq += eq2
# 	eq *= constraints_weight
# 	return eq.hamiltonian
