"""File with templates"""

import json
import logging
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Literal, get_args, overload

from qiskit.primitives.containers import SamplerPubLike

from qlauncher.base import Algorithm, Backend, Problem, Result
from qlauncher.base.base import Model
from qlauncher.problems.circuit import _Circuit
from qlauncher.routines.circuits import CIRCUIT_FORMATS
from qlauncher.routines.qiskit.algorithms.wrapper import CircuitRunner
from qlauncher.routines.qiskit.utils import coerce_to_circuit_list


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
    def __init__(
        self, problem: Problem | Model, algorithm: Algorithm, backend: Backend, /, *, logger: logging.Logger | None = None
    ) -> None:
        """
        Create a QLauncher instance that solves a `problem` using a given `algorithm` on a `backend`.

        Args:
                problem (Problem): Problem to solve.
                algorithm (Algorithm): Algorithm to use.
                backend (Backend | None, optional): Backend to run on.
                logger (logging.Logger | None, optional): Logger. Defaults to None.
        """

    @overload
    def __init__(
        self, circuit: SamplerPubLike | CIRCUIT_FORMATS, backend: Backend, /, *, shots: int = 1024, logger: logging.Logger | None = None
    ) -> None:
        """
        Create a QLauncher instance that samples `circuit` on the `backend` for `shots` shots.

        Args:
                circuit (SamplerPubLike): Circuit or (circuit, params) to sample.
                backend (Backend): Backend to run the circuit on.
                shots (int, optional): Samples to draw. Defaults to 1024.
                logger (logging.Logger | None, optional): Logger. Defaults to None.
        """

    @overload
    def __init__(self, problem: Problem | Model, algorithm: Algorithm, /, *, logger: logging.Logger | None = None) -> None:
        """
        Create a QLauncher instance that solves a `problem` using a given workflow `algorithm`. Backend is None.

        Args:
                problem (Problem | Model): Problem to solve.
                algorithm (Algorithm): Algorithm to use.
                logger (logging.Logger | None, optional): Logger. Defaults to None.
        """

    def __init__(self, *args, **kwargs) -> None:
        if len(args) == 3:
            problem: Problem | Model = args[0]
            algorithm: Algorithm = args[1]
            backend: Backend = args[2]
        elif len(args) == 2 and isinstance(args[0], Problem | Model):
            problem: Problem | Model = args[0]
            algorithm: Algorithm = args[1]
            backend: Backend = Backend('')
        elif len(args) == 2 and isinstance(args[0], (*get_args(CIRCUIT_FORMATS), *get_args(SamplerPubLike))):
            problem, algorithm, backend = self._build_from_circuit(args[0], args[1], kwargs.get('shots', 1024))
        else:
            raise TypeError
        self.problem: Problem | Model = problem
        self.algorithm = algorithm
        self.backend = backend

        logger = kwargs.get('logger')
        if logger is None:
            logger = logging.getLogger('QLauncher')
        self.logger = logger

        self.result: Result | None = None

    def _get_compatible_problem(self, **formatter_kwargs) -> Model:
        input_format = self.algorithm.get_input_format()
        if input_format is None:
            raise TypeError
        problem = self.problem
        methods = self._bfs_search(problem, input_format)
        if methods is None:
            raise TypeError

        if len(methods) == 0:
            return problem

        if isinstance(problem, Problem):
            # The first method is the Problem -> Model formatter.
            problem = methods[0](problem, **formatter_kwargs)

        for method in methods[1:]:
            problem = method(problem)

        return problem

    def run(self, **formatter_kwargs) -> Result:
        """
        Finds proper formatter, and runs the algorithm on the problem with given backends.

        Returns:
                dict: The results of the algorithm execution.
        """
        self.result = self.algorithm.run(self._get_compatible_problem(**formatter_kwargs), self.backend)
        self.logger.info('Algorithm ended successfully!')
        return self.result

    def _build_from_circuit(
        self, circuit: SamplerPubLike | CIRCUIT_FORMATS, backend: Backend, shots: int
    ) -> tuple[Model, Algorithm, Backend]:
        return (_Circuit(coerce_to_circuit_list(circuit)[0]), CircuitRunner(shots), backend)

    def _bfs_search(self, problem: Problem | Model, input_format: type[Model]) -> list[Callable[[Problem | Model], Model]] | None:
        to_check: list[tuple[list, type[Problem] | type[Model]]] = [([], type(problem))]
        visited: set = {type(problem)}
        if isinstance(problem, input_format):
            return []
        while len(to_check) > 0:
            parents, current = to_check.pop(0)
            if current is input_format:
                return parents
            for child, method in current._mapping.items():
                if isinstance(child, str):
                    child = Model._all_problems[child]
                if child in visited:
                    continue
                to_check.append((parents + [method], child))
        return None

    def save(self, path: str | Path, save_format: Literal['pickle', 'txt', 'json'] = 'pickle') -> None:
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
