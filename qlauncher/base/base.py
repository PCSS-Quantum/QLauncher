from abc import ABC, abstractmethod
import statistics
from dataclasses import dataclass
import pickle
from typing import Any, Literal
from collections.abc import Callable
import logging


AVAILABLE_FORMATS = Literal['hamiltonian', 'qubo', 'bqm', 'none', 'fn', 'tabular_ml']


@dataclass
class Result:
    data: Any

    def __str__(self):
        return f"Result holding data of type{type(self.data)}"

    def __repr__(self):
        return str(self)


@dataclass
class OptimizationResult(Result):
    bitstring_counts: dict[str, int] | dict[int, int]
    energies: dict[str, float] | dict[int, float] | list[float]

    def __str__(self) -> str:
        return f"Optimization result from {self.n_samples} samples"

    @property
    def best_bitstring(self) -> str | None:
        """Bitstring with least energy, if info is available, else None"""
        if isinstance(self.energies, dict):
            return min(self.energies, key=self.energies.get)
        return None

    @property
    def best_energy(self) -> float:
        """Lowest encountered energy"""
        if isinstance(self.energies, dict):
            return min(self.energies.values())
        return min(self.energies)

    @property
    def most_common_bitstring(self) -> str:
        """Most commonly sampled bitstring"""
        return max(self.bitstring_counts, key=self.bitstring_counts.get)

    @property
    def most_common_bitstring_energy(self) -> float | None:
        """Energy of most commonly sampled bitstring, if available, else None"""
        if isinstance(self.energies, dict):
            return self.energies[self.most_common_bitstring]
        return None

    @property
    def average_energy(self) -> float:
        """
        Average energy. If energy per bitstring is available, 
        then the number of bitstring occurrences is taken into account.
        """
        if isinstance(self.energies, dict):
            return statistics.mean([c*self.energies[bs] for bs, c in self.energies.items()])
        return statistics.mean(self.energies)

    @property
    def energy_std(self) -> float:
        """
        Standard deviation of energies. If energy per bitstring is available, 
        then the number of bitstring occurrences is taken into account.
        """
        if isinstance(self.energies, dict):
            mean = self.average_energy
            std = 0
            for bitstring, occ in self.bitstring_counts.items():
                std += occ * ((self.energies[bitstring] - mean)**2)
            return (std/(self.n_samples-1))**0.5
        return statistics.stdev(self.energies)

    @property
    def n_samples(self):
        """Total samples taken."""
        return sum(self.bitstring_counts.values())


class Backend:
    """
    Abstract class representing a backend for quantum computing.

    Attributes:
        name (str): The name of the backend.
        path (str | None): The path to the backend (optional).
        parameters (list): A list of parameters for the backend (optional).

    """

    def __init__(self, name: str, parameters: list | None = None) -> None:
        self.name: str = name
        self.is_device = name == 'device'
        self.path: str | None = None
        self.parameters = parameters if parameters is not None else []
        self.logger: logging.Logger | None = None

    def set_logger(self, logger: logging.Logger):
        self.logger = logger

    def _get_path(self):
        return f'{self.name}'


class Problem(ABC):
    """
    Abstract class for defining Problems.

    Attributes:
        variant (str): The variant of the problem. The default variant is "Optimization".
        path (str | None): The path to the problem.
        name (str): The name of the problem.
        instance_name (str): The name of the instance.
        instance (any): An instance of the problem.

    """

    _problem_id = None

    def __init__(self, instance: Any, instance_name: str = 'unnamed') -> None:
        """
        Initializes a Problem instance.

        Params:
            instance (any): An instance of the problem.
            instance_name (str | None): The name of the instance.

        Returns:
            None
        """
        self.instance: Any = instance
        self.instance_name = instance_name
        self.variant: str = 'Optimization'
        self.path: str | None = None
        self.name = self.__class__.__name__.lower()

    @classmethod
    def from_file(cls: type['Problem'], path: str) -> 'Problem':
        with open(path, 'rb') as f:
            instance = pickle.load(f)
        return cls(instance)

    @staticmethod
    def from_preset(instance_name: str, **kwargs):
        raise NotImplementedError()

    def __init_subclass__(cls) -> None:
        if Problem not in cls.__bases__:
            return
        cls._problem_id = cls

    def read_result(self, exp, log_path):
        """
        Reads a result from a file.

        Args:
            exp: The experiment.
            log_path: The path to the log file.

        Returns:
            The result.
        """
        exp += exp  # ?: this is perplexing
        with open(log_path, 'rb') as file:
            res = pickle.load(file)
        return res

    def analyze_result(self, result) -> Any:
        """
        Analyzes the result.

        Args:
            result: The result.

        """
        raise NotImplementedError()


class Algorithm(ABC):
    """
    Abstract class for Algorithms.

    Attributes:
        name (str): The name of the algorithm, derived from the class name in lowercase.
        path (str | None): The path to the algorithm, if applicable.
        parameters (list): A list of parameters for the algorithm.
        alg_kwargs (dict): Additional keyword arguments for the algorithm.

    Abstract methods:
        __init__(self, **alg_kwargs): Initializes the Algorithm object.
        _get_path(self) -> str: Returns the common path for the algorithm.
        run(self, problem: Problem, backend: Backend): Runs the algorithm on a specific problem using a backend.
    """
    _algorithm_format: AVAILABLE_FORMATS = 'none'

    def __init__(self, **alg_kwargs) -> None:
        self.name: str = self.__class__.__name__.lower()
        self.path: str | None = None
        self.parameters: list = []
        self.alg_kwargs = alg_kwargs

    def parse_result_to_json(self, o: object) -> dict:
        """Parses results so that they can be saved as a JSON file.

        Args:
            o (object): The result object to be parsed.

        Returns:
            dict: The parsed result as a dictionary.
        """
        print('Algorithm does not have the parse_result_to_json method implemented')
        return o.__dict__

    @abstractmethod
    def run(self, problem: Problem, backend: Backend, formatter: Callable) -> Result:
        """Runs the algorithm on a specific problem using a backend.

        Args:
            problem (Problem): The problem to be solved.
            backend (Backend): The backend to be used for execution.
        """
