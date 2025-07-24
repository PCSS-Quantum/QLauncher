"""Wrapper for QLauncher that enables the user to launch tasks asynchronously (futures + multiprocessing)"""
from typing import Literal
from collections.abc import Callable
import time

from qlauncher.problems import Raw
from qlauncher.base.base import Backend, Algorithm, Problem, Result
from qlauncher.launcher.qlauncher import QLauncher
from qlauncher.launcher.aql.aql_task import AQLTask, get_timeout


class AQL:
    """
    Launches QLauncher task asynchronously.

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
        self.mode: Literal['default', 'optimize_session'] = mode

        self._classical_tasks: list[AQLTask] = []
        self._quantum_tasks: list[AQLTask] = []

    def running_task_count(self) -> int:
        """
        Returns:
            int: Amount of tasks that are currently executing.
        """
        return sum(1 for t in self.tasks if t.running())

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
        except Exception as e:
            self.cancel_running_tasks()
            raise e

    def add_task(
        self,
        launcher: QLauncher | tuple[Problem, Algorithm, Backend],
        dependencies: list[AQLTask] | None = None,
        callbacks: list[Callable] | None = None,
        **kwargs
    ) -> AQLTask:
        """
        Add a QLauncher task to the execution queue.

        Args:
            launcher (QLauncher | tuple[Problem, Algorithm, Backend]): Launcher instance that will be run.
            dependencies (list[AQLTask] | None, optional): Tasks that must finish first before this task. Defaults to None.
            callbacks (list[Callable] | None, optional): 
                    Functions to run when the task finishes. 
                    The task will be passed to the function as the only parameter. 
                    Defaults to None.

        Returns:
            AQLTask: Pointer to the submitted task.
        """
        if isinstance(launcher, tuple):
            launcher = QLauncher(*launcher)

        dependencies_list = dependencies if dependencies is not None else []

        if self.mode != 'optimize_session' or not launcher.backend.is_device:
            task = AQLTask(
                lambda: launcher.run(**kwargs),
                dependencies=dependencies,
                callbacks=(callbacks if callbacks is not None else [])
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
            dependencies=[dep for dep in dependencies_list if not dep in self._quantum_tasks]
        )

        def quantum_task(formatted, *_):
            ql = QLauncher(
                Raw(formatted, launcher.problem.instance_name),
                launcher.algorithm,
                launcher.backend
            )
            return ql.run()

        t_quant = AQLTask(
            quantum_task,
            dependencies=[t_gen] + [dep for dep in dependencies_list if dep in self._quantum_tasks],
            callbacks=(callbacks if callbacks is not None else []),
            pipe_dependencies=True  # Receive output from formatter
        )

        self._classical_tasks.append(t_gen)
        self._quantum_tasks.append(t_quant)
        self.tasks.append(t_quant)

        return t_quant

    def start(self):
        """Start tasks execution."""
        for t in self._classical_tasks+self._quantum_tasks:
            if t.cancelled() or t.done() or t.running():
                raise ValueError("Cannot start again, some tasks were already ran or are currently running.")

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
            lambda: True,
            dependencies=list(quantum_dependencies)
        )

        for qt in self._quantum_tasks:
            qt.dependencies.append(gateway_task_classical)

        gateway_task_quantum = AQLTask(
            lambda: 42,
            dependencies=self._quantum_tasks.copy()
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for t in self.tasks:
            if t.running():
                t.cancel()
