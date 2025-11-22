from bisect import bisect_right
from typing import Literal

from qiskit.quantum_info import SparsePauliOp

import qlauncher.hampy as hampy

from .scheduler import JobShopScheduler, KeyList, get_label


def get_jss_hamiltonian(job_dict, max_time, onehot, version: Literal['decision', 'optimization']) -> SparsePauliOp:
	scheduler = QiskitScheduler(job_dict, max_time, onehot)
	return scheduler.get_hamiltonian(version)


class QiskitScheduler(JobShopScheduler):
	def __init__(self, job_dict, max_time=None, onehot='exact'):
		super().__init__(job_dict, max_time)
		self.H_pos_by_label = {}
		self.H_label_by_pos = {}
		self.onehot = onehot

	def _add_one_start_constraint(self, lagrange_one_hot=1) -> None:
		for task in self.tasks:
			task_times = {get_label(task, t) for t in range(self.max_time)}
			onehot_tasks = set()
			for label in task_times:
				if label in self.absurd_times:
					continue
				onehot_tasks.add(self.H_pos_by_label[label])
			if self.onehot == 'exact':
				self.H += (~hampy.one_in_n(onehot_tasks, self.n)).hamiltonian
			elif self.onehot == 'quadratic':
				self.H += hampy.one_in_n(onehot_tasks, self.n, quadratic=True).hamiltonian

	def _add_precedence_constraint(self, lagrange_precedence=1) -> None:
		for current_task, next_task in zip(self.tasks, self.tasks[1:]):
			if current_task.job != next_task.job:
				continue
			for t in range(self.max_time):
				current_label = get_label(current_task, t)
				if current_label in self.absurd_times:
					continue
				var1 = self.H_pos_by_label[current_label]
				for tt in range(min(t + current_task.duration, self.max_time)):
					next_label = get_label(next_task, tt)
					if next_label in self.absurd_times:
						continue
					var2 = self.H_pos_by_label[next_label]
					equation = hampy.Equation(self.n)
					self.H += (equation[var1] & equation[var2]).hamiltonian

	def _add_share_machine_constraint(self, lagrange_share=1) -> None:
		sorted_tasks = sorted(self.tasks, key=lambda x: x.machine)
		wrapped_tasks = KeyList(sorted_tasks, lambda x: x.machine)

		head = 0
		while head < len(sorted_tasks):
			tail = bisect_right(wrapped_tasks, sorted_tasks[head].machine)
			same_machine_tasks = sorted_tasks[head:tail]

			head = tail

			if len(same_machine_tasks) < 2:
				continue

			for task in same_machine_tasks:
				for other_task in same_machine_tasks:
					if task.job == other_task.job and task.position == other_task.position:
						continue

					for t in range(self.max_time):
						current_label = get_label(task, t)
						if current_label in self.absurd_times:
							continue

						var1 = self.H_pos_by_label[current_label]

						for tt in range(t, min(t + task.duration, self.max_time)):
							this_label = get_label(other_task, tt)
							if this_label in self.absurd_times:
								continue
							var2 = self.H_pos_by_label[this_label]
							equation = hampy.Equation(self.n)
							self.H += (equation[var1] & equation[var2]).hamiltonian

	def _build_variable_dict(self) -> None:
		for task in self.tasks:
			task_times = {get_label(task, t) for t in range(self.max_time)}
			for label in task_times:
				if label in self.absurd_times:
					continue
				self.H_pos_by_label[label] = len(self.H_pos_by_label)
				self.H_label_by_pos[len(self.H_label_by_pos)] = label
		self.n = len(self.H_pos_by_label)

	def get_hamiltonian(self, version: Literal['decision', 'optimization'] = 'optimization') -> SparsePauliOp:
		self._remove_absurd_times({}, {}, [])
		self._build_variable_dict()
		self._add_one_start_constraint()
		self._add_precedence_constraint()
		self._add_share_machine_constraint()
		# Get BQM
		# bqm = dwavebinarycsp.stitch(self.csp, **stitch_kwargs)

		# Edit BQM to encourage the shortest schedule
		# Overview of this added penalty:
		# - Want any-optimal-schedule-penalty < any-non-optimal-schedule-penalty
		# - Suppose there are N tasks that need to be scheduled and N > 0
		# - Suppose the the optimal end time for this schedule is t_N
		# - Then the worst optimal schedule would be if ALL the tasks ended at time t_N. (Since
		#   the optimal schedule is only dependent on when the LAST task is run, it is irrelevant
		#   when the first N-1 tasks end.) Note that by "worst" optimal schedule, I am merely
		#   referring to the most heavily penalized optimal schedule.
		#
		# Show math satisfies any-optimal-schedule-penalty < any-non-optimal-schedule-penalty:
		# - Penalty scheme. Each task is given the penalty: base^(task-end-time). The penalty
		#   of the entire schedule is the sum of penalties of these chosen tasks.
		# - Chose the base of my geometric series to be N+1. This simplifies the math and it will
		#   become apparent why it's handy later on.
		#
		# - Comparing the SUM of penalties between any optimal schedule (on left) with that of the
		#   WORST optimal schedule (on right). As shown below, in this penalty scheme, any optimal
		#   schedule penalty <= the worst optimal schedule.
		#     sum_i (N+1)^t_i <= N * (N+1)^t_N, where t_i the time when the task i ends  [eq 1]
		#
		# - Now let's show that all optimal schedule penalties < any non-optimal schedule penalty.
		#   We can prove this by applying eq 1 and simply proving that the worst optimal schedule
		#   penalty (below, on left) is always less than any non-optimal schedule penalty.
		#     N * (N+1)^t_N < (N+1)^(t_N + 1)
		#                               Note: t_N + 1 is the smallest end time for a non-optimal
		#                                     schedule. Hence, if t_N' is the end time of the last
		#                                     task of a non-optimal schedule, t_N + 1 <= t_N'
		#                   <= (N+1)^t_N'
		#                   < sum^(N-1) (N+1)^t_i' + (N+1)^t_N'
		#                   = sum^N (N+1)^t_i'
		#                               Note: sum^N (N+1)^t' is the sum of penalties for a
		#                                     non-optimal schedule
		#
		# - Therefore, with this penalty scheme, all optimal solution penalties < any non-optimal
		#   solution penalties

		# Get BQM
		if version == 'decision':
			return self.H.simplify().copy()
		# Get our pruned (remove_absurd_times) variable list so we don't undo pruning
		# pruned_variables = list(bqm.variables)

		for i in self.last_task_indices:
			task = self.tasks[i]

			for t in range(self.max_time):
				end_time = t + task.duration

				# Check task's end time; do not add in absurd times
				if end_time > self.max_time:
					continue

				label = get_label(task, t)
				if label in self.absurd_times:
					continue

				var = self.H_pos_by_label[label]
				self.H += hampy.Variable(var, hampy.Equation(self.n)).to_equation().hamiltonian
		# self.H += h / (base * len(self.last_task_indices))

		# Get BQM
		return self.H.simplify().copy()
