""" File with templates """
from typing import Literal
import json
import pickle
import logging
from qlauncher.base.adapter_structure import get_formatter, ProblemFormatter
from qlauncher.base import Problem, Algorithm, Backend, Result
from qlauncher.problems import Raw


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

    def __init__(self, problem: Problem, algorithm: Algorithm, backend: Backend | None = None, logger: logging.Logger | None = None) -> None:

        if not isinstance(problem, Problem):
            problem = Raw(problem)

        self.problem: Problem = problem
        self.algorithm: Algorithm = algorithm
        self.backend: Backend | None = backend
        self.formatter: ProblemFormatter = get_formatter(self.problem._problem_id, self.algorithm._algorithm_format)

        if logger is None:
            logger = logging.getLogger('QLauncher')
        self.logger = logger

        self.result: Result | None = None

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
            raise ValueError("No result to save")

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
            raise ValueError(
                f'format: {save_format} in not supported try: pickle, txt, csv or json')


def fix_json(o: object):
    # if o.__class__.__name__ == 'SamplingVQEResult':
    #     parsed = self.algorithm.parse_samplingVQEResult(o, self._full_path)
    #     return parsed
    if o.__class__.__name__ == 'complex128':
        return repr(o)
    print(
        f'Name of object {o.__class__} not known, returning None as a json encodable')
    return None
