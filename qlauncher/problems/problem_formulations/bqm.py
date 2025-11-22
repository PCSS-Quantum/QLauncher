import ast

import numpy as np
from pyqubo import Spin

from qlauncher.base import adapter, formatter
from qlauncher.base.problem_like import BQM, QUBO
from qlauncher.problems.problem_initialization import Raw


@adapter('qubo', 'bqm')
def qubo_to_bqm(qubo: QUBO) -> BQM:
	bqm, _ = QUBOMatrix(qubo.matrix, qubo.offset).qubo_matrix_into_bqm()
	return bqm


class QUBOMatrix:
	def __init__(self, matrix: np.ndarray, offset: float):
		self.matrix = matrix
		self.offset = offset

		self.symetric = (self.matrix.transpose() == self.matrix).all()
		if not self.symetric:
			for i in range(len(self.matrix)):
				for j in range(len(self.matrix)):
					if i > j:
						self.matrix[i][j] = 0

	def _get_values_and_qubits(self, matrix):
		"""
		Function to get values and qubits from matrix in form of dictionary
		where keys are indexes of qubits and values are values of matrix
		The function does not take into account zeros in matrix
		Example:
		matrix = [[0,1,2],[1,0,3],[2,3,0]]
		result = {(0,1):1, (0,2):2, (1,0):1, (1,2):3, (2,0):2, (2,1):3}
		Function also return second value which is the number of qubits
		"""
		result = {(x, y): c for y, r in enumerate(matrix) for x, c in enumerate(r) if c != 0}
		return result, len(matrix)

	def qubo_matrix_into_bqm(self):
		values_and_qubits = {(x, y): c for y, r in enumerate(self.matrix) for x, c in enumerate(r) if c != 0}
		number_of_qubits = len(self.matrix)
		qubits = [Spin(f'x{i}') for i in range(number_of_qubits)]
		H = 0
		for (x, y), value in values_and_qubits.items():
			if self.symetric:
				H += value / len({x, y}) * qubits[x] * qubits[y]
			else:
				H += value * qubits[x] * qubits[y]
		model = H.compile()
		bqm = model.to_bqm()
		bqm.offset += self.offset
		return bqm, model


@formatter(Raw, 'bqm')
def Rawbqm(problem: Raw):
	return problem.instance


def to_bqm(self: QUBO) -> BQM:
	matrix = self.matrix
	symmetric = (self.matrix.transpose() == self.matrix).all()
	if not symmetric:
		for i in range(len(self.matrix)):
			for j in range(len(self.matrix)):
				if i > j:
					matrix[i][j] = 0

	values_and_qubits = {(x, y): c for y, r in enumerate(matrix) for x, c in enumerate(r) if c != 0}
	number_of_qubits = len(matrix)
	qubits = [Spin(f'x{i}') for i in range(number_of_qubits)]
	H = 0
	for (x, y), value in values_and_qubits.items():
		if symmetric:
			H += value / len({x, y}) * qubits[x] * qubits[y]
		else:
			H += value * qubits[x] * qubits[y]
	model = H.compile()
	bqm = model.to_bqm()
	bqm.offset += self.offset
	return BQM(bqm, model)
