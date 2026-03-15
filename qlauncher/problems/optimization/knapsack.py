"""This module contains the Knapsack class."""

from collections.abc import Sequence
from typing import Literal

import numpy as np
from pyqubo import Array, Binary

from qlauncher import models
from qlauncher.base import Problem


class Knapsack(Problem):
    """
    Class for 0/1 Knapsack Problem.

    This class represents the 0/1 Knapsack problem: maximize the total value of chosen items
    subject to a capacity constraint on total weight. The class wraps a concrete instance
    and can be passed into QLauncher.
    """

    def __init__(self, values: Sequence[int], weights: Sequence[int], capacity: int, instance_name: str = 'unnamed'):
        if len(values) != len(weights) or len(values) == 0:
            raise ValueError('values and weights must have the same positive length')
        if capacity < 0:
            raise ValueError('capacity must be non-negative')
        super().__init__((values, weights, capacity), instance_name=instance_name)
        self.values = values
        self.weights = weights
        self.capacity = capacity

    @staticmethod
    def from_preset(instance_name: Literal['default', 'small'], **kwargs) -> 'Knapsack':
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
                raise ValueError(f'Preset f{instance_name} not defined')
        return Knapsack(values, weights, capacity, instance_name)

    def to_bqm(self, penalty_weight: float = 2.0, value_weight: float = 1.0) -> models.BQM:
        """
        Returns BQM for Knapsack problem.
        """
        size = len(self.values)

        x = Array.create('a_x', shape=size, vartype='BINARY')

        m = 1 if self.capacity == 0 else int(np.ceil(np.log2(self.capacity + 1)))
        y = Array.create('z_y', shape=m, vartype='BINARY')
        slack = sum((2**k) * y[k] for k in range(m))

        weight_sum = sum(self.weights[i] * x[i] for i in range(size))
        penalty = weight_sum + slack - self.capacity
        penalty *= penalty
        value_term = sum(self.values[i] * x[i] for i in range(size))
        H: Binary = penalty_weight * penalty - value_weight * value_term
        return models.BQM(H.compile())
