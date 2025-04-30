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

        class _Raw(Raw):
            pass

        formatter(_Raw, problem_type)(_raw_formatter)
        Raw.__cached_classes__[problem_type] = _Raw
        return _Raw

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
        raise ValueError

    def __new__(cls, obj: Any, instance_name: str = 'Raw'):
        if cls is not Raw:
            return super().__new__(cls)
        problem_type = cls._auto_map_problem(obj)
        true_cls = cls._cache_class(problem_type)
        return true_cls(obj, instance_name)


def _raw_formatter(raw: Raw) -> Any:
    return raw.instance
