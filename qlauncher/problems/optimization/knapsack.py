"""This module contains the Knapsack class."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from pyqubo import Array, Binary

from qlauncher.base import Problem
from qlauncher.base.problem_like import QUBO


@dataclass(frozen=True)
class KnapsackInstance:
	values: Sequence[int]
	weights: Sequence[int]
	capacity: int


class Knapsack(Problem):
	"""
	Class for 0/1 Knapsack Problem.

	This class represents the 0/1 Knapsack problem: maximize the total value of chosen items
	subject to a capacity constraint on total weight. The class wraps a concrete instance
	and can be passed into QLauncher.
	"""

	def __init__(self, instance: KnapsackInstance, instance_name: str = "unnamed"):
		if len(instance.values) != len(instance.weights) or len(instance.values) == 0:
			raise ValueError('values and weights must have the same positive length')
		if instance.capacity < 0:
			raise ValueError('capacity must be non-negative')
		super().__init__(instance, instance_name=instance_name)

	@staticmethod
	def from_lists(values: Sequence[int], weights: Sequence[int], capacity: int, name: str = 'unnamed') -> 'Knapsack':
		return Knapsack(KnapsackInstance(values=list(values), weights=list(weights), capacity=int(capacity)), instance_name=name)

	@staticmethod
	def from_preset(instance_name: str) -> 'Knapsack':
		values, weights, capacity = None, None, None
		match instance_name:
			case 'default':
				values = [9, 6, 7, 5]
				weights = [6, 4, 5, 3]
				capacity = 9
			case 'small':
				values = [4, 3, 2]
				weights = [3, 2, 2]
				capacity = 5
			case _:
				raise ValueError(f"Preset f{instance_name} not defined")
		return Knapsack.from_lists(values=values, weights=weights, capacity=capacity, name=instance_name)

	@property
	def n(self) -> int:
		return len(self.instance.values)

	@property
	def capacity(self) -> int:
		return self.instance.capacity

	@property
	def values(self) -> list[int]:
		return list(self.instance.values)

	@property
	def weights(self) -> list[int]:
		return list(self.instance.weights)

	def to_qubo(self, penalty_weight: float = 2.0, value_weight: float = 1.0) -> QUBO:
		"""
		Returns QUBO function for Knapsack problem.
		"""
		values = self.values
		weights = self.weights
		n = len(values)

		x = Array.create('a_x', shape=n, vartype='BINARY')

		m = 1 if self.capacity == 0 else int(np.ceil(np.log2(self.capacity + 1)))
		y = Array.create('z_y', shape=m, vartype='BINARY')
		slack = sum((2**k) * y[k] for k in range(m))

		weight_sum = sum(weights[i] * x[i] for i in range(n))
		penalty = weight_sum + slack - self.capacity
		penalty *= penalty
		value_term = sum(values[i] * x[i] for i in range(n))
		H: Binary = penalty_weight * penalty - value_weight * value_term

		qubo_dict, offset = H.compile().to_qubo()
		var_labels = [f'z_y[{k}]' for k in range(m)] + [f'a_x[{i}]' for i in reversed(range(n))]
		N = len(var_labels)
		Q = np.zeros((N, N))
		for i, vi in enumerate(var_labels):
			for j, vj in enumerate(var_labels):
				key = (vi, vj)
				if key in qubo_dict:
					Q[i, j] = qubo_dict[key]

		return QUBO(Q, float(offset))
