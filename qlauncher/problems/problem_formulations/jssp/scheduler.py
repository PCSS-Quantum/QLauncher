from dataclasses import dataclass
from typing import Literal

from pyqubo import Array

from qlauncher import hampy


@dataclass
class Task:
	uid: int
	job: str
	position: int
	machine: str
	duration: int

	def __repr__(self):
		return ('{{job: {job}, position: {position}, machine: {machine}, duration: {duration}}}').format(**vars(self))

	def __hash__(self) -> int:
		return self.uid.__hash__()


class JobShopScheduler:
	tasks: list[Task]
	tasks_by_machine: dict[str, set[Task]]
	tasks_by_job: dict[str, list[Task]]
	max_time: int

	def __init__(
		self,
		job_dict: dict[str, list[tuple[str, int]]],
		max_time: int | None = None,
		variant: Literal['qubo', 'hamiltonian'] = 'qubo',
		onehot: Literal['exact', 'quadratic'] | None = None,
	):
		self.tasks = []
		self.tasks_by_machine = {}
		self.tasks_by_job = {}
		self.valid_assignments = set()
		self.variant = variant
		self._process_data(job_dict, max_time)
		self._prepare_valid_assignments()
		self.n = len(self.valid_assignments)
		if self.variant == 'qubo':
			self.array = Array.create('variables', self.n, vartype='BINARY')
			self.qubo = 0
		else:
			self.equation = hampy.Equation(self.n)
			self.onehot = onehot

	def _process_data(self, jobs: dict[str, list[tuple[str, int]]], max_time: int | None = None) -> None:
		tasks = []
		last_task_indices = [-1]
		total_time = 0

		for job_name, job_tasks in jobs.items():
			last_task_indices.append(last_task_indices[-1] + len(job_tasks))

			for i, (machine, time_span) in enumerate(job_tasks):
				task = Task(len(tasks), job_name, i, machine, time_span)
				tasks.append(task)
				self.tasks_by_job[job_name].append(task)
				self.tasks_by_machine[machine].add(task)
				total_time += time_span

		self.tasks = tasks

		if max_time is not None:
			self.max_time = max_time
		else:
			self.max_time = total_time

	def _prepare_valid_assignments(self) -> None:
		for task in self.tasks:
			for t in range(self.max_time):
				self.valid_assignments.add((task, t))
		self._exclude_invalid()
		self.assignment_index = {}
		i = 0
		for task in self.tasks:
			for t in range(self.max_time):
				if self.valid(task, t):
					self.assignment_index[(task, t)] = i
					i += 1

	def _exclude_invalid(self) -> None:
		for tasks in self.tasks_by_job.values():
			predecessor_time = 0
			for task in tasks:
				for t in range(predecessor_time):
					self.valid_assignments.remove((task, t))
				predecessor_time += task.duration

			successor_time = -1
			for task in tasks[::-1]:
				successor_time += task.duration
				for t in range(successor_time):
					self.valid_assignments.remove((task, t))

	def _exclude_on_demand(
		self, disable_till: dict[str, int], disable_since: dict[str, int], disabled_variables: list[tuple[str, int, int]]
	) -> None:
		for task in self.tasks:
			if task.machine in disable_till:
				for i in range(disable_till[task.machine]):
					self.valid_assignments.remove((task, i))
			elif task.machine in disable_since:
				for i in range(disable_since[task.machine], self.max_time):
					self.valid_assignments.remove((task, i))

		for job, position, time in disabled_variables:
			task = next(filter(lambda x: x.job == job and x.position == position, self.tasks))
			self.valid_assignments.remove((task, time))

	def valid(self, task: Task, time: int) -> bool:
		return (task, time) in self.valid_assignments

	def _add_one_start_constraint(self, lagrange_one_hot: float = 1) -> None:
		"""self.csp gets the constraint: A task can start once and only once"""
		for task in self.tasks:
			if self.variant == 'qubo':
				qubo_term = 0
			else:
				onehot_tasks = set()

			for t in range(self.max_time):
				if not self.valid(task, t):
					continue

				if self.variant == 'qubo':
					qubo_term += self.array[self.assignment_index[(task, t)]]
				else:
					onehot_tasks.add(self.assignment_index[(task, t)])

			if self.variant == 'qubo':
				self.qubo += lagrange_one_hot * ((1 - qubo_term) * (1 - qubo_term))
			else:
				if self.onehot == 'exact':
					self.H += lagrange_one_hot * (~hampy.one_in_n(onehot_tasks, self.n)).hamiltonian
				elif self.onehot == 'quadratic':
					self.H += lagrange_one_hot * hampy.one_in_n(onehot_tasks, self.n, quadratic=True).hamiltonian

	def _add_precedence_constraint(self, lagrange_precedence: float = 1) -> None:
		"""self.csp gets the constraint: Task must follow a particular order.
		Note: assumes self.tasks are sorted by jobs and then by position
		"""
		for tasks in self.tasks_by_job.values():
			for current_task, next_task in zip(tasks, tasks[1:], strict=False):
				# Forming constraints with the relevant times of the next task
				for t in range(self.max_time):
					if not self.valid(current_task, t):
						continue
					var1 = (
						self.array[self.assignment_index[(current_task, t)]]
						if self.variant == 'qubo'
						else self.equation[self.assignment_index[(current_task, t)]]
					)
					for tt in range(min(t + current_task.duration, self.max_time)):
						if not self.valid(next_task, tt):
							continue
						var2 = (
							self.array[self.assignment_index[(next_task, tt)]]
							if self.variant == 'qubo'
							else self.equation[self.assignment_index[(next_task, tt)]]
						)
						if self.variant == 'qubo':
							self.qubo += lagrange_precedence * var1 * var2
						else:
							self.H += lagrange_precedence * (var1 & var2).hamiltonian

	def _add_share_machine_constraint(self, lagrange_share: float = 1) -> None:
		"""self.csp gets the constraint: At most one task per machine per time unit"""
		for tasks in self.tasks_by_machine.values():
			# Apply constraint between all tasks for each unit of time
			for task1 in tasks:
				for task2 in tasks - {task1}:
					for t in range(self.max_time):
						if not self.valid(task1, t):
							continue
						var1 = (
							self.array[self.assignment_index[(task1, t)]]
							if self.variant == 'qubo'
							else self.equation[self.assignment_index[(task1, t)]]
						)
						for tt in range(t, min(t + task1.duration, self.max_time)):
							if not self.valid(task2, tt):
								continue
							var2 = (
								self.array[self.assignment_index[(task2, tt)]]
								if self.variant == 'qubo'
								else self.equation[self.assignment_index[(task2, tt)]]
							)
							if self.variant == 'qubo':
								self.qubo += lagrange_share * var1 * var2
							else:
								self.H += lagrange_share * (var1 & var2).hamiltonian
