from abc import ABC, abstractmethod
from dataclasses import dataclass
import pickle
from typing import Any, Callable, Dict, Optional, Literal, TypeVar, overload
import logging


AVAILABLE_FORMATS = Literal['hamiltonian', 'qubo', 'none', 'fn']


@dataclass
class Result:
    best_bitstring: str
    best_energy: float
    most_common_bitstring: str
    most_common_bitstring_energy: float
    distribution: dict
    energies: dict
    num_of_samples: int
    average_energy: float
    energy_std: float
    result: Any

    def __str__(self):
        return f"Result(bitstring={self.best_bitstring}, energy={self.best_energy})"

    def __repr__(self):
        return str(self)

    def best(self):
        return self.best_bitstring, self.best_energy

    def most_common(self):
        return self.most_common_bitstring, self.most_common_bitstring_energy

    @staticmethod
    def from_distributions(bitstring_distribution: Dict[str, float], energy_distribution: Dict[str, float], result: Optional[Any] = None) -> "Result":
        """ Constructs the Result object from Dictionary with bitstring to num of occurrences, dictionary mapping bitstring to energy and optional result (rest) """
        best_bitstring = min(energy_distribution, key=energy_distribution.get)
        best_energy = energy_distribution[best_bitstring]
        most_common_bitstring = max(
            bitstring_distribution, key=bitstring_distribution.get)
        most_common_bitstring_energy = energy_distribution[most_common_bitstring]
        num_of_samples = sum(bitstring_distribution.values())

        mean_value = sum(energy_distribution[bitstring] * occ for bitstring,
                         occ in bitstring_distribution.items()) / num_of_samples
        std = 0
        for bitstring, occ in bitstring_distribution.items():
            std += occ * ((energy_distribution[bitstring] - mean_value)**2)
        std = (std/(num_of_samples-1))**0.5
        return Result(best_bitstring, best_energy, most_common_bitstring, most_common_bitstring_energy,
                      bitstring_distribution, energy_distribution, num_of_samples, mean_value, std, result)


class Backend:
    """
    Abstract class representing a backend for quantum computing.

    Attributes:
        name (str): The name of the backend.
        path (str | None): The path to the backend (optional).
        parameters (list): A list of parameters for the backend (optional).

    Methods:
        _get_path(): Returns the path to the backend.

    """

    def __init__(self, name: str, parameters: list = None) -> None:
        self.name: str = name
        self.path: str | None = None
        self.parameters = parameters if parameters is not None else []
        self.logger: Optional[logging.Logger] = None

    def set_logger(self, logger: logging.Logger):
        self.logger = logger

    def _get_path(self):
        return f'{self.name}'


P = TypeVar('P', bound='Problem')


class Problem(ABC):
    """
    Abstract class for defining Problems.

    Attributes:
        variant (str): The variant of the problem. The default variant is "Optimization".
        path (str | None): The path to the problem.
        name (str): The name of the problem.
        instance_name (str): The name of the instance.
        instance (any): An instance of the problem.

    Methods:
        set_instance(self, instance: any, instance_name: str | None = None) -> None:
            Sets the instance of the problem.
        read_instance(self, instance_path: str) -> None:
            Reads the instance of the problem from a file.
        read_result(self, exp, log_path) -> any:
            Reads a result from a file.
        analyze_result(self, result) -> None:
            Analyzes the result.

    """

    _problem_id = None

    def __init__(self, instance: Any, instance_name: str = 'unnamed') -> None:
        """
        Initializes a Problem instance.

        Args:
            instance (any): An instance of the problem.
            instance_name (str | None): The name of the instance.

        Returns:
            None
        """
        self.instance: any = instance
        self.instance_name = instance_name
        self.variant: str = 'Optimization'
        self.path: str | None = None
        self.name = self.__class__.__name__.lower()

    @classmethod
    def from_file(cls: P, path: str) -> P:
        with open(path, 'rb') as f:
            instance = pickle.load(f)
        return cls(instance)

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
        exp += exp
        with open(log_path, 'rb') as file:
            res = pickle.load(file)
        return res

    def analyze_result(self, result):
        """
        Analyzes the result.

        Args:
            result: The result.

        """


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

    @abstractmethod
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
        return dict(o)

    @abstractmethod
    def run(self, problem: Problem, backend: Backend, formatter: Callable = None) -> Result:
        """Runs the algorithm on a specific problem using a backend.

        Args:
            problem (Problem): The problem to be solved.
            backend (Backend): The backend to be used for execution.
        """
