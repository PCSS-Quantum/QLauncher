from typing import Literal

from qiskit.quantum_info import SparsePauliOp

from qlauncher import hampy
from qlauncher.hampy import Variable

from .scheduler import JobShopScheduler, Task


class HamPyScheduler(JobShopScheduler):
	def __init__(self, job_dict: dict, max_time: int | None = None, onehot: Literal['exact', 'quadratic'] = 'exact'):
		super().__init__(job_dict, max_time)
		self.equation = hampy.Equation(self.n)
		self.onehot = onehot
		self.H = 0

	def _get_variable(self, task: Task, time: int) -> Variable:
		return self.equation[self.assignment_index[(task, time)]]

	def _add_expression(self, var1: Variable, var2: Variable, lagrange_factor: float) -> None:
		self.H += lagrange_factor * (var1 & var2).hamiltonian

	def _add_expression_one_start(self, variables: list[int | Variable], lagrange_factor: float) -> None:
		if self.onehot == 'exact':
			self.H += lagrange_factor * (~hampy.one_in_n(variables, self.n)).hamiltonian
		elif self.onehot == 'quadratic':
			self.H += lagrange_factor * hampy.one_in_n(variables, self.n, quadratic=True).hamiltonian

	def get_hamiltonian(
		self,
		lagrange_one_hot: float,
		lagrange_precedence: float,
		lagrange_share: float,
		version: Literal['decision', 'optimization'] = 'optimization',
	) -> SparsePauliOp:
		self._add_one_start_constraint(lagrange_one_hot)
		self._add_precedence_constraint(lagrange_precedence)
		self._add_share_machine_constraint(lagrange_share)
		assert isinstance(self.H, SparsePauliOp)

		# Get BQM
		if version == 'decision':
			return self.H.simplify().copy()

		for tasks in self.tasks_by_job.values():
			task = tasks[-1]

			for t in range(self.max_time):
				if not self.valid(task, t):
					continue

				var = self._get_variable(task, t)
				self.H += var.to_equation().hamiltonian

		# Get BQM
		return self.H.simplify().copy()
