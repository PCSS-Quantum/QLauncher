"""Module for Job Shop Scheduling Problem (JSSP)."""

from collections import defaultdict
from typing import Literal

import numpy as np
from dimod import BinaryQuadraticModel

from qlauncher.base import Problem
from qlauncher.base.problem_like import BQM, QUBO, Hamiltonian
from qlauncher.problems.optimization.jssp_utils import HamPyScheduler, PyQuboScheduler


class JSSP(Problem):
	"""
	Class for Job Shop Scheduling Problem.

	This class represents Job Shop Scheduling Problem (JSSP) which is a combinatorial optimization problem that involves
	scheduling a set of jobs on a set of machines. Each job consists of a sequence of operations that must be performed
	on different machines. The objective is to find a schedule that minimizes the makespan, i.e., the total time required
	to complete all jobs. The class contains an instance of the problem, so it can be passed into QLauncher.


	Attributes:
		max_time (int): The maximum time for the scheduling problem.
		onehot (str): The one-hot encoding method to be used.
		optimization_problem (bool): Flag indicating whether the problem is an optimization problem or a decision problem.
		results (dict): Dictionary to store the results of the problem instance.

	"""

	def __init__(
		self,
		max_time: int,
		instance: dict[str, list[tuple[str, int]]],
		instance_name: str = 'unnamed',
		optimization_problem: bool = False,
	) -> None:
		super().__init__(instance=instance, instance_name=instance_name)
		self.max_time = max_time
		self.optimization_problem = optimization_problem
		self.variant: Literal['decision', 'optimization'] = 'optimization' if optimization_problem else 'decision'

	@property
	def setup(self) -> dict:
		return {
			'jobs': self.instance,
			'max_time': self.max_time,
			'optimization_problem': self.optimization_problem,
			'instance_name': self.instance_name,
		}

	def _get_path(self) -> str:
		return f'{self.name}@{self.instance_name}@{self.max_time}@{"optimization" if self.optimization_problem else "decision"}'

	@staticmethod
	def from_preset(instance_name: str, **kwargs) -> 'JSSP':
		match instance_name:
			case 'default':
				max_time = 3
				instance = {'cupcakes': [('mixer', 2), ('oven', 1)], 'smoothie': [('mixer', 1)], 'lasagna': [('oven', 2)]}
			case _:
				raise ValueError(f"Instance {instance_name} does not exist choose instance_name from the following: ('toy')")

		return JSSP(max_time=max_time, instance=instance, instance_name=instance_name, **kwargs)

	@classmethod
	def from_file(cls, path: str, **kwargs) -> 'JSSP':
		job_dict = defaultdict(list)
		with open(path, encoding='utf-8') as file_:
			file_.readline()
			for i, line in enumerate(file_):
				lint = list(map(int, line.split()))
				job_dict[i + 1] = list(
					zip(
						lint[::2],  # machines
						lint[1::2],  # operation lengths
						strict=False,
					)
				)
		return JSSP(instance=job_dict, **kwargs)

	def to_qubo(
		self,
		lagrange_one_hot: float = 1,
		lagrange_precedence: float = 2,
		lagrange_share: float = 5,
	) -> QUBO:
		# Define the matrix Q used for QUBO
		bqm = self._to_dimod_bqm(lagrange_one_hot, lagrange_precedence, lagrange_share)
		linear = bqm.spin.linear
		quadratic = bqm.spin.quadratic
		variables = list(linear.keys())
		self.instance_size = len(variables)
		reverse_dict_map = {v: i for i, v in enumerate(variables)}

		Q = np.zeros((self.instance_size, self.instance_size))

		for (label_i, label_j), value in quadratic.items():
			i = reverse_dict_map[label_i]
			j = reverse_dict_map[label_j]
			Q[i, j] += value
			Q[j, i] = Q[i, j]

		for label_i, value in linear.items():
			i = reverse_dict_map[label_i]
			Q[i, i] += value
		return QUBO(Q / max(np.max(Q), -np.min(Q)), 0)

	def to_hamiltonian(
		self,
		lagrange_one_hot: float = 1,
		lagrange_precedence: float = 2,
		lagrange_share: float = 5,
		onehot: Literal['exact', 'quadratic'] = 'exact',
	) -> Hamiltonian:
		scheduler = HamPyScheduler(self.instance, self.max_time, onehot)
		return Hamiltonian(scheduler.get_hamiltonian(lagrange_one_hot, lagrange_precedence, lagrange_share, self.variant).simplify())

	def _to_dimod_bqm(
		self,
		lagrange_one_hot: float,
		lagrange_precedence: float,
		lagrange_share: float,
	) -> BinaryQuadraticModel:
		scheduler = PyQuboScheduler(self.instance, self.max_time)
		return scheduler.get_bqm(lagrange_one_hot, lagrange_precedence, lagrange_share)

	def to_bqm(
		self,
		lagrange_one_hot: float = 1,
		lagrange_precedence: float = 2,
		lagrange_share: float = 5,
	) -> BQM:
		return BQM(self._to_dimod_bqm(lagrange_one_hot, lagrange_precedence, lagrange_share))
