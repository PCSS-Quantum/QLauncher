from dimod import BinaryQuadraticModel
from pyqubo import Array, Binary

from .scheduler import JobShopScheduler, Task


class PyQuboScheduler(JobShopScheduler):
	array: Array
	qubo: Binary

	def __init__(self, job_dict: dict, max_time: int | None = None):
		super().__init__(job_dict, max_time)
		self.array = Array.create('variables', self.n, vartype='BINARY')
		self.qubo = 0

	def _get_variable(self, task: Task, time: int) -> Binary:
		return self.array[self.assignment_index[(task, time)]]

	def _add_expression(self, var1: Binary, var2: Binary, lagrange_factor: float) -> None:
		self.qubo += lagrange_factor * var1 * var2

	def _add_expression_one_start(self, variables: list[Binary], lagrange_factor: float) -> None:
		self.qubo += lagrange_factor * ((1 - sum(variables)) * (1 - sum(variables)))

	def _add_variable(self, var: Binary, bias: float) -> None:
		self.qubo += var * bias

	def _get_final(self) -> BinaryQuadraticModel:
		return self.qubo.compile().to_bqm()
