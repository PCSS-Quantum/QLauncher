"""File with templates"""

import json
import logging
import pickle
from collections.abc import Callable
from typing import Literal

from qlauncher.base import Algorithm, Backend, Problem, Result
from qlauncher.base.base import ProblemLike


def _extract_args(argtypes: list[tuple[str, type]], args, kwargs) -> dict[str, object]:
	if len(args) > len(argtypes):
		return {}
	as_kwargs = []
	for name, _ in argtypes[len(args) :]:
		if name not in kwargs:
			return {}
		as_kwargs.append(kwargs[name])

	result = {}

	for expected, received in zip(argtypes, list(args) + as_kwargs):
		name, wanted_type = expected
		if not isinstance(received, wanted_type):
			return {}
		result[name] = received

	return result


class QLauncher:
	"""
	QLauncher class.

	Qlauncher is used to run quantum algorithms on specific problem instances and backends.
	It provides methods for binding parameters, preparing the problem, running the algorithm, and processing the results.

	Attributes:
		problem (Problem): The problem instance to be solved.
		algorithm (Algorithm): The quantum algorithm to be executed.
		backend (Backend, optional): The backend to be used for execution. Defaults to None.
		path (str): The path to save the results. Defaults to 'results/'.
		binding_params (dict or None): The parameters to be bound to the problem and algorithm. Defaults to None.
		encoding_type (type): The encoding type to be used changing the class of the problem. Defaults to None.

	Example of usage::

	        from qlauncher import QLauncher
	        from qlauncher.problems import MaxCut
	        from qlauncher.routines.qiskit import QAOA, QiskitBackend

	        problem = MaxCut(instance_name='default')
	        algorithm = QAOA()
	        backend = QiskitBackend('local_simulator')

	        launcher = QLauncher(problem, algorithm, backend)
	        result = launcher.process(save_pickle=True)
	        print(result)

	"""

	# @overload
	# def __init__(self, problem: Problem, algorithm: Algorithm, backend: Backend, logger: logging.Logger | None = None) -> None:
	# 	"""
	# 	Create a QLauncher instance that solves a `problem` using a given `algorithm` on a `backend`.

	# 	Args:
	# 		problem (Problem): Problem to solve.
	# 		algorithm (Algorithm): Algorithm to use.
	# 		backend (Backend | None, optional): Backend to run on.
	# 		logger (logging.Logger | None, optional): Logger. Defaults to None.
	# 	"""

	# @overload
	# def __init__(self, circuit: SamplerPubLike, backend: Backend, shots: int = 1024, logger: logging.Logger | None = None) -> None:
	# 	"""
	# 	Create a QLauncher instance that samples `circuit` on the `backend` for `shots` shots.

	# 	Args:
	# 		circuit (SamplerPubLike): Circuit or (circuit, params) to sample.
	# 		backend (Backend): Backend to run the circuit on.
	# 		shots (int, optional): Samples to draw. Defaults to 1024.
	# 		logger (logging.Logger | None, optional): Logger. Defaults to None.
	# 	"""

	# @overload
	# def __init__(self, problem: Problem, algorithm: Algorithm, logger: logging.Logger | None = None) -> None:
	# 	"""
	# 	Create a QLauncher instance that solves a `problem` using a given workflow `algorithm`. Backend is None.

	# 	Args:
	# 		problem (Problem): Problem to solve.
	# 		algorithm (Algorithm): Algorithm to use.
	# 		logger (logging.Logger | None, optional): Logger. Defaults to None.
	# 	"""

	def __init__(
		self,
		problem: Problem | ProblemLike,
		algorithm: Algorithm[ProblemLike, Backend],
		backend: Backend,
		logger: logging.Logger | None = None,
	) -> None:
		args_matched = False
		self.problem = problem
		self.algorithm = algorithm
		self.backend = backend
		# for init_set in [
		# 	# Standard Problem, Algorithm, Backend
		# 	([('problem', object), ('algorithm', Algorithm), ('backend', Backend)], self._build_from_PAB),
		# 	# Circuit running
		# 	([('circuit', object), ('backend', Backend)], self._build_from_circuit),
		# 	# Workflows: Problem, Algorithm (workflow)
		# 	([('problem', object), ('algorithm', Algorithm)], lambda parse: self._build_from_PAB(parse | {'backend': None})),
		# ]:
		# 	arg_set, build_function = init_set
		# 	parse = _extract_args(arg_set, args, kwargs)
		# 	if not parse:
		# 		continue

		# 	args_matched = True
		# 	build_function(parse | kwargs)
		# 	break

		# if not args_matched:
		# 	raise ValueError(
		# 		'Incorrect argument set to create a QLauncher instance! Expected either (Problem, Algorithm, Backend), (Problem, Algorithm) or (Qiskit sampler pub like [for example Quantum Circuit], Backend)'
		# 	)

		# logger = kwargs.get('logger')

		if logger is None:
			logger = logging.getLogger('QLauncher')
		self.logger = logger

		self.result: Result | None = None
		self._plugins = []

	def run(self, **kwargs) -> Result:
		"""
		Finds proper formatter, and runs the algorithm on the problem with given backends.

		Returns:
			dict: The results of the algorithm execution.
		"""
		input_format = self.algorithm.get_input_format()
		if input_format is None:
			raise TypeError
		problem = self.problem
		methods = self._bfs_search(problem, input_format)
		if methods is None:
			raise TypeError
		for method in methods:
			problem = method(problem)

		self.result = self.algorithm.run(problem, self.backend)
		self.logger.info('Algorithm ended successfully!')
		return self.result

	def _bfs_search(
		self, problem: Problem | ProblemLike, input_format: type[ProblemLike]
	) -> list[Callable[[Problem | ProblemLike], ProblemLike]] | None:
		to_check: list[tuple[list, type[Problem] | type[ProblemLike]]] = [([], type(problem))]
		visited: set = {type(problem)}
		if isinstance(problem, input_format):
			return []
		while len(to_check) > 0:
			parents, current = to_check.pop(0)
			if current is input_format:
				return parents
			for child, method in current._mapping.items():
				if isinstance(child, str):
					child = ProblemLike._all_problems[child]
				if child in visited:
					continue
				to_check.append((parents + [method], child))
		return None

	def save(self, path: str, save_format: Literal['pickle', 'txt', 'json'] = 'pickle') -> None:
		"""
		Save last run result to file

		Args:
			path (str): File path.
			save_format (Literal[&#39;pickle&#39;, &#39;txt&#39;, &#39;json&#39;], optional): Save format. Defaults to 'pickle'.

		Raises:
			ValueError: When no result is available or an incorrect save format was chosen
		"""
		if self.result is None:
			raise ValueError('No result to save')

		# if not os.path.isfile(path):
		#     path = os.path.join(path, f'result-{datetime.now().isoformat(sep="_").replace(":","_")}.{save_format}')

		self.logger.info('Saving results to file: %s', str(path))
		if save_format == 'pickle':
			with open(path, mode='wb') as f:
				pickle.dump(self.result, f)
		elif save_format == 'json':
			with open(path, mode='w', encoding='utf-8') as f:
				json.dump(self.result.__dict__, f, default=fix_json)
		elif save_format == 'txt':
			with open(path, mode='w', encoding='utf-8') as f:
				f.write(str(self.result))
		else:
			raise ValueError(f'format: {save_format} in not supported try: pickle, txt, csv or json')

	# def get_plugins(self, plugin_type: WHEN) -> list[Plugin]:
	# 	def condition_check(plugin: Plugin) -> bool:
	# 		return plugin._when == plugin_type

	# 	return list(filter(condition_check, self._plugins))

	# @property
	# def plugins(self) -> list[Plugin]:
	# 	return self._plugins

	# def add_plugin(self, plugin: Plugin) -> None:
	# 	self._plugins.append(plugin)


def fix_json(o: object):
	# if o.__class__.__name__ == 'SamplingVQEResult':
	#     parsed = self.algorithm.parse_samplingVQEResult(o, self._full_path)
	#     return parsed
	if o.__class__.__name__ == 'complex128':
		return repr(o)
	print(f'Name of object {o.__class__} not known, returning None as a json encodable')
	return None
