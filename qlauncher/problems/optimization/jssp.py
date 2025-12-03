"""Module for Job Shop Scheduling Problem (JSSP)."""

from collections import defaultdict
from typing import Literal

from qiskit.quantum_info import SparsePauliOp

from qlauncher.base import Problem
from qlauncher.base.problem_like import BQM, Hamiltonian
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
	def from_preset(instance_name: Literal['default'], **kwargs) -> 'JSSP':
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

	def to_bqm(
		self,
		lagrange_one_hot: float = 1,
		lagrange_precedence: float = 2,
		lagrange_share: float = 5,
	) -> BQM:
		# Define the matrix Q used for QUBO
		scheduler = PyQuboScheduler(self.instance, self.max_time)
		result = scheduler.get_result(lagrange_one_hot, lagrange_precedence, lagrange_share)
		if isinstance(result, SparsePauliOp):
			raise TypeError
		return BQM(result)

	def to_hamiltonian(
		self,
		lagrange_one_hot: float = 1,
		lagrange_precedence: float = 2,
		lagrange_share: float = 5,
		onehot: Literal['exact', 'quadratic'] = 'exact',
	) -> Hamiltonian:
		scheduler = HamPyScheduler(self.instance, self.max_time, onehot)
		result = scheduler.get_result(lagrange_one_hot, lagrange_precedence, lagrange_share, self.variant)
		if not isinstance(result, SparsePauliOp):
			raise TypeError
		return Hamiltonian(result)
