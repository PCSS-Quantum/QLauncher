"""File with templates"""

import json
import logging
import pickle
from typing import Literal, overload

from qiskit.primitives.containers import SamplerPubLike

from qlauncher.base import Algorithm, Backend, Problem, Result
from qlauncher.base.adapter_structure import ProblemFormatter, get_formatter
from qlauncher.problems import Raw, _Circuit
from qlauncher.routines.qiskit.algorithms.wrapper import _CircuitRunner


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

	@overload
	def __init__(self, problem: Problem, algorithm: Algorithm, backend: Backend, logger: logging.Logger | None = None) -> None:
		"""
		Create a QLauncher instance that solves a `problem` using a given `algorithm` on a `backend`.

		Args:
		    problem (Problem): Problem to solve.
		    algorithm (Algorithm): Algorithm to use.
		    backend (Backend | None, optional): Backend to run on.
		    logger (logging.Logger | None, optional): Logger. Defaults to None.
		"""

	@overload
	def __init__(self, circuit: SamplerPubLike, backend: Backend, shots: int = 1024, logger: logging.Logger | None = None) -> None:
		"""
		Create a QLauncher instance that samples `circuit` on the `backend` for `shots` shots.

		Args:
		    circuit (SamplerPubLike): Circuit or (circuit, params) to sample.
		    backend (Backend): Backend to run the circuit on.
		    shots (int, optional): Samples to draw. Defaults to 1024.
		    logger (logging.Logger | None, optional): Logger. Defaults to None.
		"""

	@overload
	def __init__(self, problem: Problem, algorithm: Algorithm, logger: logging.Logger | None = None) -> None:
		"""
		Create a QLauncher instance that solves a `problem` using a given workflow `algorithm`. Backend is None.

		Args:
		    problem (Problem): Problem to solve.
		    algorithm (Algorithm): Algorithm to use.
		    logger (logging.Logger | None, optional): Logger. Defaults to None.
		"""

	def __init__(self, *args, **kwargs) -> None:
		args_matched = False
		for init_set in [
			# Standard Problem, Algorithm, Backend
			([('problem', object), ('algorithm', Algorithm), ('backend', Backend)], self._build_from_PAB),
			# Circuit running
			([('circuit', object), ('backend', Backend)], self._build_from_circuit),
			# Workflows: Problem, Algorithm (workflow)
			([('problem', object), ('algorithm', Algorithm)], lambda parse: self._build_from_PAB(parse | {'backend': None})),
		]:
			arg_set, build_function = init_set
			parse = _extract_args(arg_set, args, kwargs)
			if not parse:
				continue

			args_matched = True
			build_function(parse | kwargs)
			break

		if not args_matched:
			raise ValueError(
				'Incorrect argument set to create a QLauncher instance! Expected either (Problem, Algorithm, Backend), (Problem, Algorithm) or (Qiskit sampler pub like [for example Quantum Circuit], Backend)'
			)

		self.formatter: ProblemFormatter = get_formatter(self.problem._problem_id, self.algorithm._algorithm_format)

		logger = kwargs.get('logger')

		if logger is None:
			logger = logging.getLogger('QLauncher')
		self.logger = logger

		self.result: Result | None = None

	def _build_from_circuit(self, parsed: dict):
		self.problem = _Circuit(parsed['circuit'], parsed.get('shots', 1024))
		self.algorithm = _CircuitRunner()
		self.backend: Backend = parsed['backend']

	def _build_from_PAB(self, parsed: dict):
		if not isinstance(parsed['problem'], Problem):
			self.problem = Raw(parsed['problem'])
		else:
			self.problem: Problem = parsed['problem']

		self.algorithm = parsed['algorithm']
		self.backend: Backend = parsed['backend']

	def run(self, **kwargs) -> Result:
		"""
		Finds proper formatter, and runs the algorithm on the problem with given backends.

		Returns:
		    dict: The results of the algorithm execution.
		"""

		self.formatter.set_run_params(kwargs)

		self.result = self.algorithm.run(self.problem, self.backend, formatter=self.formatter)
		self.logger.info('Algorithm ended successfully!')
		return self.result

	def save(self, path: str, save_format: Literal['pickle', 'txt', 'json'] = 'pickle'):
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


def fix_json(o: object):
	# if o.__class__.__name__ == 'SamplingVQEResult':
	#     parsed = self.algorithm.parse_samplingVQEResult(o, self._full_path)
	#     return parsed
	if o.__class__.__name__ == 'complex128':
		return repr(o)
	print(f'Name of object {o.__class__} not known, returning None as a json encodable')
	return None
