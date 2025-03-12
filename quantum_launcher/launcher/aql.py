"""Wrapper for QuantumLauncher that enables the user to launch tasks asynchronously (futures + multiprocessing)"""
from typing import Tuple, Literal
from collections.abc import Callable
from concurrent import futures
from threading import Event

from pathos.multiprocessing import ProcessingPool

from quantum_launcher.base.base import Backend, Algorithm, Problem, Result
from quantum_launcher.launcher.qlauncher import QuantumLauncher
from quantum_launcher.problems import Raw


class AQLTask:
    """
        Task object returned to user, so that dependencies can be created. Basically a Future wrapper, that doesn't start immediately and can have other tasks it depends on.

        Attributes:
            task (Callable): function that gets executed asynchronously
            executor (futures.Executor): Executor to use for submitting jobs.
            dependencies (list[AQLTask]): Optional dependencies. The task will wait for all its dependencies to finish, before starting.
            callbacks (list[Callable]): Callbacks ran when the task finishes executing.
            pipe_dependencies (bool): If True results of tasks defined as dependencies will be passed as arguments to self.task. Defaults to False.
    """

    def __init__(
        self,
        task: Callable,
        executor: futures.Executor,
        dependencies: list['AQLTask'] | None = None,
        callbacks: list[Callable] | None = None,
        pipe_dependencies: bool = False
    ) -> None:

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

    def start(self):
        """Start task execution."""
        self._set_future()

    def cancel(self) -> bool:
        """
        Attempt to cancel the task.

        Returns:
            bool: True if cancellation was successful
        """
        if self._future is None:
            return False
        return self._future.cancel()

    def cancelled(self) -> bool:
        """
        Returns:
            bool: True if the task was cancelled by the user.
        """
        if self._future is None:
            return False
        return self._future.cancelled()

    def done(self) -> bool:
        """
        Returns:
            bool: True if the task had finished execution.
        """
        if self._future is None:
            return False
        return self._future.done()

    def running(self) -> bool:
        """
        Returns:
            bool: True if the task is currently executing.
        """
        if self._future is None:
            return False
        return self._future.running()

    def result(self, timeout: float | int | None = None) -> Result:
        """
        Get result of running the task.
        Blocks the thread until task is finished.

        Args:
            timeout (float | int | None, optional): The maximum amount to wait for execution to finish. If None, wait forever. If not None and time runs out, raises TimeoutError. Defaults to None.
        Returns:
            Result
        """
        self._future_made.wait(timeout=timeout)  # Wait until we submit a task to the executor
        return self._future.result(timeout=timeout)


class AQL:
    """
    Launches QuantumLauncher task asynchronously.

    Attributes:
        tasks (list[AQLTask]): list of submitted tasks.
        mode (Literal[&#39;default&#39;, &#39;optimize_session&#39;]): Execution mode.

    Usage Example
    -------------
    ::

        aql = AQL(mode='optimize_session')

        t_real = aql.add_task(
            (
                GraphColoring.from_preset('small'), 
                QAOA(),
                AQTBackend(token='valid_token', name='device')
            ),
            constraints_weight=5,
            costs_weight=1
            )

        aql.add_task(
            (
                GraphColoring.from_preset('small'), 
                QAOA(),
                QiskitBackend('local_simulator')
            ),
            dependencies=[t_real],
            constraints_weight=5,
            costs_weight=1
        )

        aql.start()
        result_real, result_sim = aql.results(timeout=15)


    """

    def __init__(
        self,
        mode: Literal['default', 'optimize_session'] = 'default'
    ) -> None:
        """
        Args:
            mode (Literal[&#39;default&#39;, &#39;optimize_session&#39;], optional): Task execution mode. If 'optimize_session' all tasks running on a real quantum device get split into separate generation and run subtasks, then the quantum tasks are ran in one shorter block. Defaults to 'default'.
        """

        self.tasks: list[AQLTask] = []
        self.mode = mode

        self._classical_tasks = []
        self._quantum_tasks = []

        self._executor = futures.ThreadPoolExecutor()

    def wait_for_finish(self, timeout: float | int | None = None) -> None:
        """
        Blocks the thread until all tasks are finished.

        Args:
            timeout (float | int | None, optional): The maximum amount to wait for execution to finish. If None, wait forever. If not None and time runs out, raises TimeoutError. Defaults to None.
        """
        self.results(timeout=timeout)

    def running_feature_count(self) -> int:
        """
        Returns:
            int: Amount of tasks that are currently executing.
        """
        return len([t for t in self.tasks if t.running()])

    def results(self, timeout: float | int | None = None) -> list[Result | None]:
        """
        Get a list of results from tasks.
        Results are ordered in the same way the tasks were added.
        Blocks the thread until all tasks are finished.

        Args:
            timeout (float | int | None, optional): The maximum amount to wait for execution to finish. If None, wait forever. If not None and time runs out, raises TimeoutError. Defaults to None.

        Returns:
            list[Result | None]: Task results.
        """
        return [t.result(timeout=timeout) if not t.cancelled() else None for t in self.tasks]

    def add_task(self, launcher: QuantumLauncher | Tuple[Problem, Algorithm, Backend], dependencies: list[AQLTask] | None = None, callbacks: list[Callable] | None = None, **kwargs) -> AQLTask:
        """
        Add a Quantum launcher task to the execution queue.

        Args:
            launcher (QuantumLauncher | Tuple[Problem, Algorithm, Backend]): Launcher instance that will be run.
            dependencies (list[AQLTask] | None, optional): Tasks that must finish first before this task. Defaults to None.
            callbacks (list[Callable] | None, optional): Functions to run when the task finishes. The task will be passed to the function as the only parameter. Defaults to None.

        Returns:
            AQLTask: Pointer to the submitted task.
        """
        if isinstance(launcher, tuple):
            launcher = QuantumLauncher(*launcher)

        dependencies_list = dependencies if dependencies is not None else []

        if self.mode != 'optimize_session' or not launcher.backend.is_device:
            task = AQLTask(
                lambda: launcher.run(**kwargs),
                dependencies=dependencies,
                callbacks=(callbacks if callbacks is not None else []),
                executor=self._executor
            )
            self.tasks.append(task)
            self._classical_tasks.append(task)
            return task

        def gen_task():
            launcher.formatter.set_run_params(kwargs)
            return launcher.formatter(launcher.problem)

        # Split real device task into generation and actual run on a QC
        t_gen = AQLTask(
            gen_task,
            dependencies=[dep for dep in dependencies_list if not dep in self._quantum_tasks],
            executor=self._executor
        )

        def quantum_task(formatted, *rest):
            ql = QuantumLauncher(
                Raw(formatted, launcher.problem.instance_name),
                launcher.algorithm,
                launcher.backend
            )
            return ql.run()

        t_quant = AQLTask(
            quantum_task,
            dependencies=[t_gen] + [dep for dep in dependencies_list if dep in self._quantum_tasks],
            callbacks=(callbacks if callbacks is not None else []),
            pipe_dependencies=True,  # Receive output from formatter
            executor=self._executor)

        self._classical_tasks.append(t_gen)
        self._quantum_tasks.append(t_quant)
        self.tasks.append(t_quant)

        return t_quant

    def start(self):
        """Start tasks execution."""
        self._run_async()

    def _run_async(self):
        quantum_dependencies = set()
        dependency_queue = self._quantum_tasks.copy()
        while dependency_queue:
            t = dependency_queue.pop(0)
            quantum_dependencies |= set(t.dependencies)
            dependency_queue += t.dependencies

        quantum_dependencies = quantum_dependencies.difference(self._quantum_tasks)
        # The gateway tasks will ensure execution order of (all classical tasks that quantum tasks depend on) - (all quantum tasks) - (rest of the classical tasks)
        gateway_task_classical = AQLTask(
            lambda: 42,
            dependencies=list(quantum_dependencies),
            executor=self._executor
        )

        for qt in self._quantum_tasks:
            qt.dependencies.append(gateway_task_classical)

        gateway_task_quantum = AQLTask(
            lambda: 42,
            dependencies=self._quantum_tasks.copy(),
            executor=self._executor
        )

        for ct in [t for t in self._classical_tasks if (not t in quantum_dependencies)]:
            ct.dependencies.append(gateway_task_quantum)

        self._classical_tasks.append(gateway_task_classical)
        self._quantum_tasks.append(gateway_task_quantum)

        # Start all tasks
        for t in self._classical_tasks:
            t.start()

        for qt in self._quantum_tasks:
            qt.start()
