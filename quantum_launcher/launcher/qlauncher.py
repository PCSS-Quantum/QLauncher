""" File with templates """
import json
import pickle
from typing import List, Literal, Optional, Union, Callable
from quantum_launcher.base.adapter_structure import get_formatter, ProblemFormatter
from quantum_launcher.base import Problem, Algorithm, Backend, Result
from quantum_launcher.problems import Raw
import logging


class QuantumLauncher:
    """
    Quantum Launcher class.

    Quantum launcher is used to run quantum algorithms on specific problem instances and backends.
    It provides methods for binding parameters, preparing the problem, running the algorithm, and processing the results.

    Attributes:
        problem (Problem): The problem instance to be solved.
        algorithm (Algorithm): The quantum algorithm to be executed.
        backend (Backend, optional): The backend to be used for execution. Defaults to None.
        path (str): The path to save the results. Defaults to 'results/'.
        binding_params (dict or None): The parameters to be bound to the problem and algorithm. Defaults to None.
        encoding_type (type): The encoding type to be used changing the class of the problem. Defaults to None.

    Example of usage::

            from templates import QuantumLauncher
            from problems import MaxCut
            from qiskit_routines import QAOA, QiskitBackend

            problem = MaxCut(instance_name='default')
            algorithm = QAOA()
            backend = QiskitBackend('local_simulator')

            launcher = QuantumLauncher(problem, algorithm, backend)
            result = launcher.process(save_pickle=True)
            print(result)

    """

    def __init__(self, problem: Problem, algorithm: Algorithm, backend: Backend = None,
                 logger: Optional[logging.Logger] = None) -> None:

        if not isinstance(problem, Problem):
            problem = Raw(problem)

        self.problem: Problem = problem
        self.algorithm: Algorithm = algorithm
        self.backend: Backend = backend

        if logger is None:
            logger: logging.Logger = logging.getLogger('QuantumLauncher')
        self.logger = logger

        # logging.info(f'Found proper formatter, with formatter structure: {self.formatter.get_pipeline()}')

        self.res: dict = {}

    def run(self, **kwargs) -> Result:
        """
        Finds proper formatter, and runs the algorithm on the problem with given backends.

        Returns:
            dict: The results of the algorithm execution.
        """

        formatter: ProblemFormatter = get_formatter(self.problem._problem_id, self.algorithm._algorithm_format)
        formatter.set_run_params(kwargs)

        self.result = self.algorithm.run(self.problem, self.backend, formatter=formatter)
        logging.info('Algorithm ended successfully!')
        return self.result

    def save(self, path: str, format: Literal['pickle', 'txt', 'json'] = 'pickle'):
        logging.info(f'Saving results to file: {path}')
        if format == 'pickle':
            with open(path, mode='wb') as f:
                pickle.dump(self.result, f)
        elif format == 'json':
            with open(path, mode='w', encoding='utf-8') as f:
                json.dump(self.result.__dict__, f, default=fix_json)
        elif format == 'txt':
            with open(path, mode='w', encoding='utf-8') as f:
                f.write(self.result.__str__())
        else:
            raise ValueError(
                f'format: {format} in not supported try: pickle, txt, csv or json')

    def process(self, *, file_path: Optional[str] = None, format: Union[Literal['pickle', 'txt', 'json'], List[Literal['pickle', 'txt', 'json']]] = 'pickle', **kwargs) -> dict:
        """
        Runs the algorithm, processes the data, and saves the results if specified.

        Args:
            file_path (Optional[str]): Flag indicating whether to save the results to a file. Defaults to None.
            format (Union[Literal['pickle', 'txt', 'json'], List[Literal['pickle', 'txt', 'json']]]): Format in which file should be saved. Defaults to 'pickle'

        Returns:
            dict: The processed results.
        """
        results = self.run(**kwargs)
        energy = results.result['energy']
        res = {}
        res['problem_setup'] = self.problem.setup
        res['algorithm_setup'] = self.algorithm.setup
        res['algorithm_setup']['variant'] = self.problem.variant
        res['backend_setup'] = self.backend.setup
        res['results'] = results

        self._file_name = self.problem.path + '-' + \
            self.backend.path + '-' \
            + self.algorithm.path + '-' + str(energy)

        if file_path is not None and isinstance(format, str):
            self.save(file_path, format)
        if file_path is not None and isinstance(format, list):
            for form in format:
                self.save(file_path, form)
        return res


def fix_json(o: object):
    # if o.__class__.__name__ == 'SamplingVQEResult':
    #     parsed = self.algorithm.parse_samplingVQEResult(o, self._full_path)
    #     return parsed
    if o.__class__.__name__ == 'complex128':
        return repr(o)
    print(
        f'Name of object {o.__class__} not known, returning None as a json encodable')
    return None
