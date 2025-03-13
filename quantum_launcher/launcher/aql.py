"""Wrapper for QuantumLauncher that enables the user to launch tasks asynchronously (futures + multiprocessing)"""
from typing import Tuple, Literal, Any
from collections.abc import Callable
from concurrent import futures
from threading import Event
import time
import multiprocess

from pathos.multiprocessing import _ProcessPool

from quantum_launcher.base.base import Backend, Algorithm, Problem, Result
from quantum_launcher.launcher.qlauncher import QuantumLauncher
from quantum_launcher.problems import Raw


def get_timeout(max_timeout: int | float | None, start: int | float) -> int | float | None:
    """
    Get timeout to wait on an event, useful when awaiting multiple tasks and total timeout must be max_timeout.

    Args:
        max_timeout (int | float | None): Total allowed timeout, None = infinite wait.
        start (int | float): Await start timestamp (time.time())

    Returns:
        int | float | None: Remaining timeout or None if max_timeout was None
    """
    if max_timeout is None:
        return None
    return max_timeout - (time.time() - start)


class AQLTask:
    """
        Task object returned to user, so that dependencies can be created. 
        Basically a Future wrapper, that doesn't start immediately and can have other tasks it depends on.

        Attributes:
            task (Callable): function that gets executed asynchronously
            executor (futures.Executor): Executor to use for submitting jobs.
            dependencies (list[AQLTask]): Optional dependencies. The task will wait for all its dependencies to finish, before starting.
            callbacks (list[Callable]): Callbacks ran when the task finishes executing.
            pipe_dependencies (bool): If True results of tasks defined as dependencies will be passed as arguments to self.task. 
            Defaults to False.
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

        self._executor = executor

        self._cancelled = False
        self._future = None
        self._pool = _ProcessPool(processes=1)
        self._future_made = Event()

    def _set_initial_state(self):
        self._cancelled = False
        self._future = None
        self._pool = _ProcessPool(processes=1)
        self._future_made = Event()

    def _shutdown_subprocess(self):
        self._pool.close()
        self._pool.terminate()
        self._pool.join()

    def _async_task(self) -> Any:
        dep_results = [d.result() for d in self.dependencies]

        if self._cancelled:
            return None

        res = self._pool.apply_async(self.task, args=(dep_results if self.pipe_dependencies else []))
        # Turns out you can't just outright kill threads (or futures) is so I have to do this, so that the thread knows to exit.
        while not self._cancelled:
            try:
                return res.get(timeout=0.1)
            except multiprocess.context.TimeoutError:
                pass  # task not ready, check for cancel
            # For any other error originating from the task, shutdown and clean up subprocess then raise error again.
            except Exception as e:
                self._shutdown_subprocess()
                raise e

        if self._cancelled:
            self._shutdown_subprocess()  # kill res process

        return res.get() if res.ready() else None

    def _set_future(self):
        self._future = self._executor.submit(self._async_task)
        self._future.add_done_callback(lambda fut: [cb(self) for cb in self.callbacks])  # pass AQLTask instead of future
        self._future_made.set()

    def start(self):
        """Start task execution."""
        if self._future is not None:
            return
        self._set_future()

    def cancel(self) -> bool:
        """
        Attempt to cancel the task.

        Returns:
            bool: True if cancellation was successful
        """
        self._cancelled = True  # Shutdown threaded future
        if self._future is None:
            return True
        futures.wait([self._future], 2)  # Wait for future to join
        return not self._future.running()

    def cancelled(self) -> bool:
        """
        Returns:
            bool: True if the task was cancelled by the user.
        """
        if self._future is None:
            return False
        return self._cancelled and not self._future.running()

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

    def result(self, timeout: float | int | None = None) -> Result | None:
        """
        Get result of running the task.
        Blocks the thread until task is finished.

        Args:
            timeout (float | int | None, optional): 
                    The maximum amount to wait for execution to finish.
                    If None, wait forever. If not None and time runs out, raises TimeoutError. 
                    Defaults to None.
        Returns:
            Result if future returned result or None when cancelled.
        """
        start = time.time()
        self._future_made.wait(timeout=get_timeout(timeout, start))  # Wait until we submit a task to the executor
        return self._future.result(timeout=get_timeout(timeout, start))


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
            mode (Literal[&#39;default&#39;, &#39;optimize_session&#39;], optional): 
                    Task execution mode. 
                    If 'optimize_session' all tasks running on a real quantum device get split into separate generation and run subtasks,
                    then the quantum tasks are ran in one shorter block. 
                    Defaults to 'default'.
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
            timeout (float | int | None, optional): 
                    The maximum amount to wait for execution to finish. 
                    If None, wait forever. If not None and time runs out, raises TimeoutError. 
                    Defaults to None.
        """
        self.results(timeout=timeout)

    def running_feature_count(self) -> int:
        """
        Returns:
            int: Amount of tasks that are currently executing.
        """
        return len([t for t in self.tasks if t.running()])

    def cancel_running_tasks(self):
        """Cancel all running tasks assigned to this AQL instance."""
        for t in self._classical_tasks + self._quantum_tasks:
            t.cancel()

    def results(self, timeout: float | int | None = None, cancel_tasks_on_timeout: bool = True) -> list[Result | None]:
        """
        Get a list of results from tasks.
        Results are ordered in the same way the tasks were added.
        Blocks the thread until all tasks are finished.

        Args:
            timeout (float | int | None, optional): 
                    The maximum amount to wait for execution to finish. 
                    If None, wait forever. If not None and time runs out, raises TimeoutError. 
                    Defaults to None.
            cancel_tasks_on_timeout (bool): Whether to cancel all other tasks when one task raises a TimeoutError.

        Returns:
            list[Result | None]: Task results.
        """
        try:
            start = time.time()
            return [t.result(timeout=get_timeout(timeout, start)) if not t.cancelled() else None for t in self.tasks]
        except TimeoutError as e:
            if cancel_tasks_on_timeout:
                self.cancel_running_tasks()
            raise e
        except KeyboardInterrupt:
            self.cancel_running_tasks()

    def add_task(
        self,
        launcher: QuantumLauncher | Tuple[Problem, Algorithm, Backend],
        dependencies: list[AQLTask] | None = None,
        callbacks: list[Callable] | None = None,
        **kwargs
    ) -> AQLTask:
        """
        Add a Quantum launcher task to the execution queue.

        Args:
            launcher (QuantumLauncher | Tuple[Problem, Algorithm, Backend]): Launcher instance that will be run.
            dependencies (list[AQLTask] | None, optional): Tasks that must finish first before this task. Defaults to None.
            callbacks (list[Callable] | None, optional): 
                    Functions to run when the task finishes. 
                    The task will be passed to the function as the only parameter. 
                    Defaults to None.

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
        for t in self._classical_tasks+self._quantum_tasks:
            if t.cancelled():
                raise ValueError("Cannot restart cancelled task")

        self._run_async()

    def _run_async(self):
        quantum_dependencies = set()
        dependency_queue = self._quantum_tasks.copy()
        while dependency_queue:
            t = dependency_queue.pop(0)
            quantum_dependencies |= set(t.dependencies)
            dependency_queue += t.dependencies

        quantum_dependencies = quantum_dependencies.difference(self._quantum_tasks)
        # The gateway tasks will ensure execution order of
        # (all classical tasks that quantum tasks depend on) - (all quantum tasks) - (rest of the classical tasks)
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

        for ct in [t for t in self._classical_tasks if not t in quantum_dependencies]:
            ct.dependencies.append(gateway_task_quantum)

        self._classical_tasks.append(gateway_task_classical)
        self._quantum_tasks.append(gateway_task_quantum)

        # Start all tasks
        for t in self._classical_tasks:
            t.start()

        for qt in self._quantum_tasks:
            qt.start()
