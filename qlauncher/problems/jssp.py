"""Module for Job Shop Scheduling Problem (JSSP)."""

import contextlib
from collections import defaultdict
from typing import Literal

import numpy as np

from qlauncher.base.problem_like import QUBO, Hamiltonian

with contextlib.suppress(ModuleNotFoundError):
	from qlauncher.problems.problem_formulations.jssp.qiskit_scheduler import get_jss_hamiltonian
from qlauncher.base import Problem
from qlauncher.problems.problem_formulations.jssp.pyqubo_scheduler import get_jss_bqm


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

	gamma = 1
	lagrange_one_hot = 1
	lagrange_precedence = 2
	lagrange_share = 5

	def __init__(
		self,
		max_time: int,
		instance: dict[str, list[tuple[str, int]]],
		instance_name: str = 'unnamed',
		optimization_problem: bool = False,
		onehot: Literal['exact', 'quadratic'] = 'exact',
	) -> None:
		super().__init__(instance=instance, instance_name=instance_name)
		self.max_time = max_time
		self.onehot = onehot
		self.optimization_problem = optimization_problem
		self.variant: Literal['decision', 'optimization'] = 'optimization' if optimization_problem else 'decision'
		self.onehot = onehot

	@property
	def setup(self) -> dict:
		return {
			'max_time': self.max_time,
			'onehot': self.onehot,
			'optimization_problem': self.optimization_problem,
			'instance_name': self.instance_name,
		}

	def _get_path(self) -> str:
		return (
			f'{self.name}@{self.instance_name}@{self.max_time}@{"optimization" if self.optimization_problem else "decision"}@{self.onehot}'
		)

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

	def to_hamiltonian(self) -> Hamiltonian:
		return Hamiltonian(get_jss_hamiltonian(self.instance, self.max_time, self.onehot, self.variant))

	def _fix_get_jss_bqm(self, config, lagrange_one_hot=0, lagrange_precedence=0, lagrange_share=0) -> tuple[dict, list, None]:
		pre_result = get_jss_bqm(
			self.instance,
			self.max_time,
			config,
			lagrange_one_hot=lagrange_one_hot,
			lagrange_precedence=lagrange_precedence,
			lagrange_share=lagrange_share,
		)
		result = (pre_result.spin.linear, pre_result.spin.quadratic, pre_result.spin.offset)  # I need to change it into dict somehow
		return result, list(result[0].keys()), None

	def _calculate_instance_size(self) -> int:
		# Calculate instance size for training
		_, variables, _ = self._fix_get_jss_bqm(
			self.config,
			lagrange_one_hot=self.lagrange_one_hot,
			lagrange_precedence=self.lagrange_precedence,
			lagrange_share=self.lagrange_share,
		)
		return len(variables)

	def _get_len_all_jobs(self) -> int:
		result = 0
		for job in self.instance.values():
			result += len(job)
		return result

	def _one_hot_to_jobs(self, binary_vector: list[int]) -> list[str]:
		_, variables, _ = self._fix_get_jss_bqm(
			self.config,
			lagrange_one_hot=self.lagrange_one_hot,
			lagrange_precedence=self.lagrange_precedence,
			lagrange_share=self.lagrange_share,
		)
		return [variables[i] for i in range(len(variables)) if binary_vector[i] == 1]

	def _set_config(self) -> None:
		self.config = {}
		self.config['parameters'] = {}
		self.config['parameters']['job_shop_scheduler'] = {}
		self.config['parameters']['job_shop_scheduler']['problem_version'] = 'optimization'

	def to_qubo(self) -> QUBO:
		# Define the matrix Q used for QUBO
		self.config = {}
		self.instance_size = self._calculate_instance_size()
		self._set_config()
		actually_its_qubo, variables, _ = self._fix_get_jss_bqm(
			self.config,
			lagrange_one_hot=self.lagrange_one_hot,
			lagrange_precedence=self.lagrange_precedence,
			lagrange_share=self.lagrange_share,
		)
		reverse_dict_map = {v: i for i, v in enumerate(variables)}

		Q = np.zeros((self.instance_size, self.instance_size))

		for (label_i, label_j), value in actually_its_qubo[1].items():
			i = reverse_dict_map[label_i]
			j = reverse_dict_map[label_j]
			Q[i, j] += value
			Q[j, i] = Q[i, j]

		for label_i, value in actually_its_qubo[0].items():
			i = reverse_dict_map[label_i]
			Q[i, i] += value
		return QUBO(Q / max(np.max(Q), -np.min(Q)), 0)
