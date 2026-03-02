import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

from qlauncher.base.problem_like import Model

if TYPE_CHECKING:
	from collections.abc import Callable

AVAILABLE_FORMATS = Literal['hamiltonian', 'qubo', 'bqm', 'none', 'fn', 'tabular_ml']

_Model = TypeVar('_Model', bound=Model)
_Backends = TypeVar('_Backends', bound='Backend')


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
		return f'Result(bitstring={self.best_bitstring}, energy={self.best_energy})'

	def __repr__(self):
		return str(self)

	def best(self):
		return self.best_bitstring, self.best_energy

	def most_common(self):
		return self.most_common_bitstring, self.most_common_bitstring_energy

	@staticmethod
	def from_counts_energies(bitstring_counts: dict[str, int], energies: dict[str, float], result: Any = None) -> 'Result':
		"""
		Constructs the Result object from Dictionary with bitstring to num of occurrences,
		dictionary mapping bitstring to energy and optional result (rest)
		"""
		best_bitstring = min(energies, key=energies.get)
		best_energy = energies[best_bitstring]
		most_common_bitstring = max(bitstring_counts, key=bitstring_counts.get)
		most_common_bitstring_energy = energies[most_common_bitstring]
		num_of_samples = int(sum(bitstring_counts.values()))

		mean_value = sum(energies[bitstring] * occ for bitstring, occ in bitstring_counts.items()) / num_of_samples
		std = 0
		for bitstring, occ in bitstring_counts.items():
			std += occ * ((energies[bitstring] - mean_value) ** 2)
		std = (std / (num_of_samples - 1)) ** 0.5
		return Result(
			best_bitstring,
			best_energy,
			most_common_bitstring,
			most_common_bitstring_energy,
			{k: v / num_of_samples for k, v in bitstring_counts.items()},
			energies,
			num_of_samples,
			mean_value,
			std,
			result,
		)


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

	def set_logger(self, logger: logging.Logger) -> None:
		self.logger = logger

	def _get_path(self) -> str:
		return f'{self.name}'


class Problem:
	"""
	Abstract class for defining Problems.

	Attributes:
		variant (str): The variant of the problem. The default variant is "Optimization".
		path (str | None): The path to the problem.
		name (str): The name of the problem.
		instance_name (str): The name of the instance.
		instance (any): An instance of the problem.

	"""

	_all_problems: dict[str, type['Problem']] = {}

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
		self.variant: Literal['decision', 'optimization'] = 'optimization'
		self.path: str | None = None
		self.name = self.__class__.__name__.lower()

	@classmethod
	def from_file(cls: type['Problem'], path: str) -> 'Problem':
		with open(path, 'rb') as f:
			instance = pickle.load(f)
		return cls(instance)

	@staticmethod
	def from_preset(instance_name, **kwargs) -> 'Problem':
		raise NotImplementedError()

	def __init_subclass__(cls) -> None:
		if Problem not in cls.__bases__:
			return
		Problem._all_problems[cls.__name__] = cls
		cls._mapping: dict[type[Model], Callable[[], Model]] = {}
		for method_name in cls.__dict__:
			if method_name.startswith('to_'):
				method = cls.__dict__[method_name]
				cls._mapping[method.__annotations__['return']] = method

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
			return pickle.load(file)

	def analyze_result(self, result) -> Any:
		"""
		Analyzes the result.

		Args:
			result: The result.

		"""
		raise NotImplementedError()

	def to(self, problem_type: type[Model]) -> Model:
		name = problem_type.__name__.lower()
		if hasattr(self, f'to_{name}'):
			return getattr(self, f'to_{name}')()
		raise TypeError


class OptimizationAlgorithm: ...


class Algorithm(ABC, Generic[_Model, _Backends]):
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

	@classmethod
	def get_class_input_format(cls) -> type[Model] | None:
		return cls.run.__annotations__.get('problem', None)

	def get_input_format(self) -> type[Model] | None:
		return self.get_class_input_format()

	@abstractmethod
	def run(self, problem: _Model, backend: _Backends) -> Result:
		"""Runs the algorithm on a specific problem using a backend.

		Args:
			problem (Problem): The problem to be solved.
			backend (Backend): The backend to be used for execution.
		"""
