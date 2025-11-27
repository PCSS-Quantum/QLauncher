from typing import Literal

from qiskit.quantum_info import SparsePauliOp

from qlauncher import hampy
from qlauncher.hampy import Variable

from .scheduler import JobShopScheduler, Task


class HamPyScheduler(JobShopScheduler):
	equation: hampy.Equation
	onehot: Literal['exact', 'quadratic']

	def __init__(self, job_dict: dict, max_time: int | None = None, onehot: Literal['exact', 'quadratic'] = 'exact'):
		super().__init__(job_dict, max_time)
		self.equation = hampy.Equation(self.n)
		self.onehot = onehot

	def _get_variable(self, task: Task, time: int) -> Variable:
		return self.equation[self.assignment_index[(task, time)]]

	def _add_expression(self, var1: Variable, var2: Variable, lagrange_factor: float) -> None:
		self.equation += lagrange_factor * (var1 & var2)

	def _add_expression_one_start(self, variables: list[int | Variable], lagrange_factor: float) -> None:
		if self.onehot == 'exact':
			self.equation += lagrange_factor * (~hampy.one_in_n(variables, self.n))
		elif self.onehot == 'quadratic':
			self.equation += lagrange_factor * hampy.one_in_n(variables, self.n, quadratic=True)

	def _add_variable(self, var: Variable, bias: float) -> None:
		self.equation += var.to_equation() * bias

	def _get_final(self) -> SparsePauliOp:
		return self.equation.hamiltonian.copy()
