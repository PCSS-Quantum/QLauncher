from dimod import BinaryQuadraticModel
from pyqubo import Array, Binary

from .scheduler import JobShopScheduler


class DWaveScheduler(JobShopScheduler):
	array: Array
	qubo: Binary

	def __init__(self, job_dict: dict, max_time: int | None = None):
		super().__init__(job_dict, max_time, 'qubo')

	def get_bqm(self, lagrange_one_hot: float, lagrange_precedence: float, lagrange_share: float) -> BinaryQuadraticModel:
		"""Returns a BQM to the Job Shop Scheduling problem."""

		# Apply constraints to self.csp
		self._add_one_start_constraint(lagrange_one_hot)
		self._add_precedence_constraint(lagrange_precedence)
		self._add_share_machine_constraint(lagrange_share)

		base = len(self.tasks_by_job)  # Base for exponent
		# Get our pruned (remove_absurd_times) variable list so we don't undo pruning
		# pruned_variables = list(bqm.variables)
		for tasks in self.tasks_by_job.values():
			task = tasks[-1]

			for t in range(self.max_time):
				end_time = t + task.duration

				# Add bias to variable
				bias = 2 * base ** (end_time - self.max_time)
				if not self.valid(task, t):
					continue
				var = self.array[task.uid, t]
				self.qubo += var * bias

		# Get BQM
		self.model = self.qubo.compile()
		return self.model.to_bqm()
