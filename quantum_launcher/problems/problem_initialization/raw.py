""" This module contains the Raw class."""
from typing import Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from quantum_launcher.base import Problem, formatter


class Raw(Problem):
    """ Meta class for raw problem """
    __cached_classes__: dict[str, type] = {}

    @staticmethod
    def _cache_class(problem_type: str):
        if problem_type in Raw.__cached_classes__:
            return Raw.__cached_classes__[problem_type]

        cls = type(problem_type, (Raw, ), {})
        formatter(cls, problem_type)(_raw_formatter)
        Raw.__cached_classes__[problem_type] = cls
        return cls

    @staticmethod
    def _auto_map_problem(obj: Any) -> str:
        """Automatically maps obj into str.

        Args:
            obj (Any): obj of some problem class.

        Returns:
            str: name of problem class that obj is in.
        """
        if isinstance(obj, SparsePauliOp):
            return 'hamiltonian'
        if isinstance(obj, tuple) and len(obj) == 2 and \
                isinstance(obj[0], np.ndarray) and isinstance(obj[1], (int, float)):
            return 'qubo'
        try:
            from dimod.binary.binary_quadratic_model import BinaryQuadraticModel
            if isinstance(obj, BinaryQuadraticModel):
                return 'bqm'
        except ImportError:
            pass
        return 'none'

    def __new__(cls, instance: Any, instance_name: str = 'Raw'):
        if cls is not Raw:
            return super().__new__(cls)
        problem_type = cls._auto_map_problem(instance)
        true_cls = cls._cache_class(problem_type)
        return true_cls(instance, instance_name)


def _raw_formatter(raw: Raw) -> Any:
    return raw.instance
