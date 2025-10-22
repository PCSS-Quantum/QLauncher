"""This module contains the Knapsack class."""

from collections.abc import Sequence
from dataclasses import dataclass

from qlauncher.base import Problem


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

    def __init__(self, instance: KnapsackInstance, instance_name: str | None = None):
        if len(instance.values) != len(instance.weights) or len(instance.values) == 0:
            raise ValueError('values and weights must have the same positive length')
        if instance.capacity < 0:
            raise ValueError('capacity must be non-negative')
        super().__init__(instance, instance_name=instance_name)

    @staticmethod
    def from_lists(values: Sequence[int], weights: Sequence[int], capacity: int, name: str | None = None) -> 'Knapsack':
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
                capacity = 4
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
