from typing import Iterable, Tuple, Callable, Literal
from concurrent import futures
from threading import Event
from pathos.multiprocessing import ProcessingPool

from quantum_launcher.base.base import Backend, Algorithm, Problem
from quantum_launcher.launcher.qlauncher import QuantumLauncher
from quantum_launcher.problems import Raw


class AQLTask:
    def __init__(
        self,
        task: Callable,
        dependencies: Iterable['AQLTask'] | None = None,
        callbacks: Iterable[Callable] | None = None,
        executor: futures.Executor | None = None,
        pipe_dependencies: bool = False
    ) -> None:
        """
        Task object returned to user, so that dependencies can be created. Basically a Future wrapper, that doesn't start immediately and can have other tasks it depends on.

        Args:
            task (Callable): function that gets executed asynchronously
            dependencies (Iterable[&#39;AQLTask&#39;] | None, optional): Optional dependencies. The task will wait for all its dependencies to finish, before starting. Defaults to None.
            callbacks (Iterable[Callable] | None, optional): Callbacks ran when the task finishes executing. Defaults to None.
            executor (futures.Executor | None, optional): Executor to use for submitting jobs. Defaults to None.
            pipe_dependencies (bool, optional): If True results of tasks defined as dependencies will be passed as arguments to self.task. Defaults to False.
        """

        self.task = task
        self.dependencies = dependencies if dependencies is not None else []
        self.callbacks = callbacks if callbacks is not None else []
        self.pipe_dependencies = pipe_dependencies

        self._future = None
        self._future_made = Event()
        self._executor = executor

    def _async_task(self):
        dep_results = [d.result() for d in self.dependencies]
        pool = ProcessingPool(nodes=1)
        res = pool.pipe(self.task, *(dep_results if self.pipe_dependencies else []))
        return res

    def _set_future(self):
        self._future = self._executor.submit(self._async_task)
        self._future.add_done_callback(lambda fut: [cb(self) for cb in self.callbacks])  # pass AQLTask instead of future
        self._future_made.set()

    def _start(self):
        self._set_future()

    def cancel(self) -> bool:
        if self._future is None:
            return False
        return self._future.cancel()

    def cancelled(self) -> bool:
        if self._future is None:
            return False
        return self._future.cancelled()

    def done(self) -> bool:
        if self._future is None:
            return False
        return self._future.done()

    def running(self) -> bool:
        if self._future is None:
            return False
        return self._future.running()

    def result(self, timeout=None):
        self._future_made.wait(timeout=timeout)  # Wait until we submit a task to the executor
        return self._future.result(timeout=timeout)


class AQL:
    def __init__(
        self,
        mode: Literal['default', 'optimize_session'] = 'default'
    ) -> None:

        self.tasks: list[AQLTask] = []
        self._results = []
        self._results_bitstring = []

        self.mode = mode
        self.classical_tasks = []
        self.quantum_tasks = []

        self._executor = futures.ThreadPoolExecutor()

    def wait_for_finish(self, timeout: float | int | None = None) -> None:
        r = [t.result(timeout) for t in self.tasks]

    def running_feature_count(self) -> bool:
        return len([t for t in self.tasks if t.running()])

    def get_results(self, timeout: float | int | None = None) -> tuple[list, list]:
        task_outputs = [(t.result(timeout=timeout), t.result(timeout=timeout).best_bitstring)
                        if not t.cancelled() else (None, None) for t in self.tasks]
        results, bitstrings = tuple(zip(*task_outputs))
        return list(results), list(bitstrings)

    def add_task(self, launcher: QuantumLauncher | Tuple[Problem, Algorithm, Backend], dependencies: Iterable[AQLTask] | None = None, callbacks=None) -> AQLTask:
        if isinstance(launcher, tuple):
            launcher = QuantumLauncher(*launcher)

        dependencies_list = dependencies if dependencies is not None else []

        if not self.mode == 'optimize_session' or not launcher.backend.is_device:
            # This is a solution until I think of something smarter...
            if len(set(self.quantum_tasks).intersection(set(dependencies_list))) != 0:
                raise ValueError("Quantum tasks run last, you cannot depend on them for classical tasks.")

            task = AQLTask(
                launcher.run,
                dependencies=dependencies,
                callbacks=(callbacks if callbacks is not None else []),
                executor=self._executor
            )
            self.tasks.append(task)
            self.classical_tasks.append(task)
            return task

        # Split real device task into generation and actual run on a QC
        t_gen = AQLTask(
            launcher.format,
            dependencies=[dep for dep in dependencies_list if not dep in self.quantum_tasks],
            executor=self._executor
        )

        def quant_task(formatted, *rest):
            ql = QuantumLauncher(
                Raw(formatted, launcher.problem.instance_name),
                launcher.algorithm,
                launcher.backend
            )
            return ql.run()

        t_quant = AQLTask(
            quant_task,
            dependencies=[t_gen] + [dep for dep in dependencies_list if dep in self.quantum_tasks],
            callbacks=(callbacks if callbacks is not None else []),
            pipe_dependencies=True,  # Receive output from formatter
            executor=self._executor)

        self.classical_tasks.append(t_gen)
        self.quantum_tasks.append(t_quant)
        self.tasks.append(t_quant)

        return t_quant

    def add_task_chain(self, chain: Iterable[QuantumLauncher] | Iterable[Tuple[Problem, Algorithm, Backend]]) -> AQLTask:
        """
        Add a chain of tasks that should be executed one after another.

        Args:
            chain (Iterable[QuantumLauncher] | Iterable[Tuple[Problem, Algorithm, Backend]]): Chain of tasks to execute.

        Returns:
            Last task in the chain.
        """

        if len(chain) == 0:
            raise ValueError("Chains must have at least 1 element.")

        prev_task = self.add_task(chain[0])
        for t in chain[1:]:
            new_task = self.add_task(t, dependencies=[prev_task])
            prev_task = new_task

        return prev_task

    def start(self):
        self._results = []
        self._results_bitstring = []

        self._run_async()

    def _run_async(self):
        # This task will wait for all classical tasks to execute
        gateway_task = AQLTask(
            lambda: 42,
            dependencies=self.classical_tasks,
            executor=self._executor
        )

        # Which means that all quantum tasks will execute after the main workload
        for qt in self.quantum_tasks:
            qt.dependencies.append(gateway_task)

        for t in self.classical_tasks:
            t._start()
        for qt in self.quantum_tasks:
            qt._start()
        gateway_task._start()
