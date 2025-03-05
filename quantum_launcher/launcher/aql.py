# TODO update to new QL version
from typing import Tuple
from concurrent import futures
from itertools import product


from quantum_launcher.base.base import Backend, Algorithm, Problem
from quantum_launcher.launcher.qlauncher import QuantumLauncher


class AQL:
    def __init__(
        self,
        backends: list[Tuple[Backend, int]],
        algorithms: list[Tuple[Algorithm, int]],
        problems: list[Tuple[Problem, int]],
    ) -> None:

        self.backends = backends
        self.algorithms = algorithms
        self.problems = problems
        self._results = []
        self._results_bitstring = []

        self._futures = set()
        # TODO: determine num workers (heuristic? param?)
        self._executor = futures.ThreadPoolExecutor(max_workers=42, thread_name_prefix="aql")

    def _future_done(self, f: futures.Future):
        """
        Callback added to all submitted futures. Runs when future finishes (either gets cancelled or finished)

        Args:
            f (futures.Future): future passed by futures

        Raises:
            RuntimeError: When processing a future not in self._futures (something went very wrong)
        """
        if not f in self._futures:
            raise RuntimeError(f"Attempted to finish a task that was not kept track of {f}")

        self._futures.remove(f)
        if f.cancelled():
            return

        res = f.result()
        self._results.append(res)
        self._results_bitstring.append(res.best_bitstring)

    def _launch_future(self, fn, *args, **kwargs):
        f = self._executor.submit(fn, *args, **kwargs)
        f.add_done_callback(lambda fut: self._future_done(fut))  # Lambda needed here (I assume because of some arg passing shenanigans)
        self._futures.add(f)

    def wait_for_finish(self) -> None:
        futures.wait(self._futures)

    def running_feature_count(self) -> bool:
        return len(self._futures)

    def get_results(self) -> tuple[list, list]:
        return self._results, self._results_bitstring

    def start(self):
        self._results = []
        self._results_bitstring = []

        self.run_async()

    def run_async(self):
        for back_tup, alg_tup, prob_tup in product(self.backends, self.algorithms, self.problems):
            backend, b_times = back_tup
            algorithm, a_times = alg_tup
            problem, p_times = prob_tup
            times = b_times * a_times * p_times
            for _ in range(times):
                self._launch_future(self.run_async_task, algorithm, problem, backend)

    def run_async_task(self, algorithm: Algorithm, problem: Problem, backend: Backend):
        ql = QuantumLauncher(problem, algorithm, backend)
        return ql.run()


class AQLManager:
    """
    Context manager for asyncQuantumLauncher
    Simplified high-level context manager to support asynchronous flow of asyncQuantumLauncher.

    Inside is only initialization and whole processing is done at the end.

    To save the results it's recommended to assign manager's variables to local ones, so they don't get destroyed.


    Usage Example
    -------------
    ::

        with AQLManager('my_path') as launcher:
            launcher.add()
            launcher.add()
            launcher.add()
            result = aql.result
        print(result)

    """

    def __init__(self, path: str = None):
        self.aql = None
        self.path = path
        self.result = []
        self.result_bitstring = []
        self._backends: list[Backend] = []
        self._algorithms: list[Algorithm] = []
        self._problems: list[Problem] = []

    def __enter__(self):
        return self

    def add_backend(self, backend: Backend, times: int = 1):
        self._backends.append((backend, times))

    def add_algorithm(self, algorithm, times: int = 1):
        self._algorithms.append((algorithm, times))

    def add_problem(self, problem, times: int = 1):
        self._problems.append((problem, times))

    def add(self, backend: Backend = None, algorithm: Algorithm = None, problem: Problem = None, times: int = 1):
        self._backends.append((backend, times))
        self._algorithms.append((algorithm, 1))
        self._problems.append((problem, 1))

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            raise exc_type(exc_val).with_traceback(exc_tb)
        aql = AQL(self._backends, self._algorithms, self._problems)
        aql.start()
        aql.wait_for_finish()
        result, result_bitstring = aql.get_results()
        self.result.extend(result)
        self.result_bitstring.extend(result_bitstring)


if __name__ == '__main__':
    from quantum_launcher.problems import EC
    from quantum_launcher.routines.qiskit_routines import QAOA, QiskitBackend

    with AQLManager('test') as launcher:
        launcher.add(backend=QiskitBackend('local_simulator'),
                     algorithm=QAOA(p=1), problem=EC('exact', instance_name='toy'))
        for i in range(2, 3):
            launcher.add_algorithm(QAOA(p=i))
        result = launcher.result
        result_bitstring = launcher.result_bitstring
    print(len(result))
    print(result_bitstring)

    for ind, i in enumerate(result):
        print(ind, i['SamplingVQEResult'].best_measurement)
