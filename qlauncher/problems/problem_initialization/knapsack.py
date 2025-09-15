from dataclasses import dataclass
from typing import Sequence
import numpy as np
from qlauncher.base import Problem, formatter
from pyqubo import Array

@dataclass(frozen=True)
class KnapsackInstance:
    values: Sequence[int]
    weights: Sequence[int]
    capacity: int

class Knapsack(Problem):
    """
    Problem plecakowy 0/1:
    maksymalizuj sum_i(v_i * x_i)
    przy ograniczeniu sum_i(w_i * x_i) <= C
    """
    def __init__(self, instance: KnapsackInstance, instance_name: str | None = None):
        assert len(instance.values) == len(instance.weights) and len(instance.values) > 0, \
            "values i weights muszą mieć ten sam dodatni rozmiar"
        assert instance.capacity >= 0, "capacity musi być nieujemne"
        super().__init__(instance, instance_name=instance_name)

    @staticmethod
    def from_lists(values: Sequence[int], weights: Sequence[int], capacity: int, name: str | None = None) -> "Knapsack":
        return Knapsack(KnapsackInstance(values=list(values), weights=list(weights), capacity=int(capacity)), instance_name=name)

    @staticmethod
    def from_preset(instance_name: str) -> "Knapsack":
        values, weights, capacity = None, None, None
        match instance_name:
            case 'default':
                values  = [9, 6, 7, 5]
                weights = [6, 4, 5, 3]
                capacity = 9
        return Knapsack.from_lists(values=values, weights=weights, capacity=capacity, name="default")

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

@formatter(problem=Knapsack, alg_format="qubo")
def knapsack_qubo(problem: Knapsack, penalty_weight: float = 2.0, value_weight: float = 1.0):
    """
    Formulacja QUBO z binarnymi slackami:
    Min:  A * (sum_i(w_i * x_i) + s - C)^2 - B * sum_i(v_i * x_i)
    gdzie s = sum_k(2^k * y_k) (slack),
    tak aby wymusić sum_i(w_i * x_i + s) = C.

    Zwraca: (Q_matrix: np.ndarray, offset: float)
    """
    values = problem.values
    weights = problem.weights
    C = problem.capacity
    n = len(values)

    x = Array.create("a_x", shape=n, vartype="BINARY")

    if C == 0:
        m = 1
    else:
        m = int(np.ceil(np.log2(C + 1)))
    y = Array.create("z_y", shape=m, vartype="BINARY")
    slack = sum((1 << k) * y[k] for k in range(m))

    weight_sum = sum(weights[i] * x[i] for i in range(n))
    penalty = (weight_sum + slack - C) ** 2
    value_term = sum(values[i] * x[i] for i in range(n))
    H = penalty_weight * penalty - value_weight * value_term

    qubo_dict, offset = H.compile().to_qubo()
    var_labels = [f"z_y[{k}]" for k in range(m)] + [f"a_x[{i}]" for i in reversed(range(n))]
    N = len(var_labels)
    Q = np.zeros((N, N))
    for i, vi in enumerate(var_labels):
        for j, vj in enumerate(var_labels):
            key = (vi, vj)
            if key in qubo_dict:
                Q[i, j] = qubo_dict[key]

    return Q, float(offset)
