from typing import Literal

from qiskit.quantum_info import SparsePauliOp

from .scheduler import JobShopScheduler


class QiskitScheduler(JobShopScheduler):
	def __init__(self, job_dict: dict, max_time: int | None = None, onehot: Literal['exact', 'quadratic'] = 'exact'):
		super().__init__(job_dict, max_time, "hamiltonian", onehot)

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

		# Get BQM
		if version == 'decision':
			return self.H.simplify().copy()
		# Get our pruned (remove_absurd_times) variable list so we don't undo pruning
		# pruned_variables = list(bqm.variables)

		for tasks in self.tasks_by_job.values():
			task = tasks[-1]

			for t in range(self.max_time):
				end_time = t + task.duration

				# Check task's end time; do not add in absurd times
				if end_time > self.max_time:
					continue

				if not self.valid(task,t):
					continue

				var = self.equation[self.assignment_index[(task,t)]]
				self.H += var.to_equation().hamiltonian
		# self.H += h / (base * len(self.last_task_indices))

		# Get BQM
		return self.H.simplify().copy()
