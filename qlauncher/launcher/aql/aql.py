"""Asynchronous QLauncher orchestration layer (AQL).

AQL is a lightweight orchestration utility that lets you submit one or more
`QLauncher` executions to a pluggable job manager (`BaseJobManager`) and collect the results later.

Conceptually, AQL builds a dependency graph of tasks and runs a scheduler thread that:
- submits tasks whose dependencies are satisfied,
- waits for any job to finish via the manager,
- reads results, surfaces exceptions, and triggers callbacks.

Execution modes:
- ``default``: tasks are submitted as soon as they are ready.
- ``optimize_session``: for *device* backends (``backend.is_device == True``), a task is split into a
  formatter step (problem conversion) and a quantum execution step. Two barrier "gateway" tasks are then
  inserted so that the global ordering becomes:
  (classical prerequisites of quantum) -> (all quantum tasks) -> (remaining classical tasks).

Example:
        from qlauncher.workflow.new_aql import AQL
        from qlauncher.launcher.qlauncher import QLauncher
        from qlauncher.routines.qiskit import FALQON, QiskitBackend

        problem = ...  # Problem / Hamiltonian
        algo = FALQON(max_reps=1)
        backend = QiskitBackend("local_simulator")
        launcher = QLauncher(problem, algo, backend)

        with AQL(mode="default") as aql:
                aql.add_task(launcher, shots=128)
                aql.start()
                result = aql.results(timeout=60)[0]
"""

import contextlib
import time
import weakref
from collections.abc import Callable
from threading import Event, Thread
from typing import Any, Literal

from qlauncher.base import Algorithm, Backend, Model, Problem, Result
from qlauncher.launcher.aql.aql_task import ManagerBackedTask, get_timeout
from qlauncher.launcher.qlauncher import QLauncher
from qlauncher.workflow.base_job_manager import BaseJobManager
from qlauncher.workflow.local_scheduler import LocalJobManager


def _gateway_true() -> bool:
    """
    A small barrier helper used by AQL in ``optimize_session`` mode.

    This task always returns ``True`` and is used only to enforce dependency ordering
    between groups of tasks (classical prerequisites, quantum tasks, remaining classical).

    Args:
            None.

    Returns:
            Always ``True``.
    """
    return True


def _filter_run_kwargs_for_callable(fn: Callable[..., Any], run_kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Filter ``run_kwargs`` so they are safe to forward to a callable.

    If the callable accepts ``**kwargs`` then ``run_kwargs`` is returned unchanged.
    If the callable does *not* accept ``**kwargs``, only
    keyword arguments that match the callable signature are forwarded.

    This is mainly a guardrail for calling real ``QLauncher.run(...)`` across processes,
    where tests may pass extra kwargs and where the launcher implementation may have a
    narrow signature.

    Args:
            fn: Target callable (e.g., ``QLauncher.run``).
            run_kwargs: Candidate keyword arguments to forward.

    Returns:
            A dictionary of keyword arguments that can be safely passed to ``fn``.
    """
    try:
        import inspect

        sig = inspect.signature(fn)
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            return run_kwargs
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in run_kwargs.items() if k in allowed}
    except Exception:
        return run_kwargs


class AQL:
    """
    Launches QLauncher tasks asynchronously, using a provided BaseJobManager.

    - In 'default' mode, tasks are submitted as they become ready.
    - In 'optimize_session' mode, real-device tasks are split (format + quantum run) and
            a dependency barrier is inserted so that:
            (all classical prereqs of quantum) -> (all quantum tasks) -> (remaining classical tasks).

    Notes:
    - `**run_kwargs` are forwarded to `QLauncher.run(**run_kwargs)`.
    - `manager_kwargs` are forwarded to `manager.submit(..., **manager_kwargs)`.
    """

    def __init__(
        self,
        mode: Literal['default', 'optimize_session'] = 'default',
        manager: BaseJobManager | None = None,
    ) -> None:
        """
        Create a new AQL scheduler instance.

        Args:
                mode: Scheduling mode. ``"default"`` submits tasks as soon as they are ready.
                        ``"optimize_session"`` groups device-backed quantum tasks into a single
                        serialized block by inserting barrier tasks.
                manager: Job manager used to execute tasks. If ``None``, a ``LocalJobManager`` is
                        created and used.

        Returns:
                None.
        """
        self.tasks: list[ManagerBackedTask] = []  # user-visible tasks (results order)
        self.mode: Literal['default', 'optimize_session'] = mode

        # Internal task sets (includes helper/gateway tasks)
        self._classical_tasks: list[ManagerBackedTask] = []
        self._quantum_tasks: list[ManagerBackedTask] = []

        # Per-task manager submission kwargs
        self._manager_kwargs: dict[ManagerBackedTask, dict[str, Any]] = {}

        # Scheduler state
        self._manager: BaseJobManager = manager if manager is not None else LocalJobManager()
        self._scheduler_thread: Thread | None = None
        self._scheduler_done = Event()
        self._scheduler_exc: BaseException | None = None
        self._prepared_optimize_session: bool = False

    def running_task_count(self) -> int:
        """
        Return the number of currently running internal tasks.

        Args:
                None.

        Returns:
                Number of tasks for which ``task.running()`` is ``True``.
        """
        return sum(1 for t in (self._classical_tasks + self._quantum_tasks) if t.running())

    def cancel_running_tasks(self) -> None:
        """
        Best-effort cancellation of all internal tasks (classical + quantum).

        Args:
                None.

        Returns:
                None.
        """
        for t in self._classical_tasks + self._quantum_tasks:
            with contextlib.suppress(Exception):
                t.cancel()

    def results(self, timeout: float | int | None = None, cancel_tasks_on_timeout: bool = True) -> list[Result | None]:
        """
        Collect results from user-visible tasks (in the order they were added).

        Args:
                timeout: Total timeout budget (seconds) shared across all ``result()`` waits.
                cancel_tasks_on_timeout: If ``True``, cancel any still-running tasks when a
                        ``TimeoutError`` is raised.

        Returns:
                A list of results aligned with ``add_task`` order. Cancelled tasks yield ``None``.

        Raises:
                TimeoutError: If the overall timeout is exceeded while waiting for results.
                BaseException: Re-raises any exception from the underlying task or scheduler.
        """
        try:
            start = time.time()
            out: list[Result | None] = []
            for t in self.tasks:
                out.append(t.result(timeout=get_timeout(timeout, start)) if not t.cancelled() else None)

            # If scheduler crashed, surface the error even if tasks returned None.
            if self._scheduler_exc is not None:
                raise self._scheduler_exc

            return out
        except TimeoutError as e:
            if cancel_tasks_on_timeout:
                self.cancel_running_tasks()
            raise e
        except Exception as e:
            self.cancel_running_tasks()
            raise e

    def add_task(
        self,
        launcher: QLauncher | tuple[Problem | Model, Algorithm, Backend],
        dependencies: list[ManagerBackedTask] | None = None,
        callbacks: list[Callable] | None = None,
        manager_kwargs: dict[str, Any] | None = None,
        **run_kwargs: object,
    ) -> ManagerBackedTask:
        """
        Add a QLauncher task to the execution queue.

        In ``default`` mode (or when ``launcher.backend.is_device`` is ``False``), this creates a single
        task that calls ``launcher.run(**run_kwargs)``.

        In ``optimize_session`` mode for device backends, the submission is split into:
        1) a formatter task (``launcher._get_compatible_problem``),
        2) a quantum execution task that builds a new launcher in the worker and runs it.
        The returned (user-visible) task is the quantum execution task.

        Args:
                launcher: Either a ``QLauncher`` instance or a tuple ``(problem, algorithm, backend)`` used to
                        construct one.
                dependencies: Optional list of tasks that must complete before this task can run.
                callbacks: Optional list of callables invoked with the task outcome.
                manager_kwargs: Keyword arguments forwarded to ``manager.submit(...)`` when the task is submitted.
                **run_kwargs: Keyword arguments forwarded to ``QLauncher.run(**run_kwargs)``.

        Returns:
                A ``ManagerBackedTask`` representing the submitted work. For device backends in
                ``optimize_session`` mode, this is the quantum execution task.
        """
        if isinstance(launcher, tuple):
            launcher = QLauncher(*launcher)

        deps = dependencies if dependencies is not None else []
        cb = callbacks if callbacks is not None else []
        mkwargs = manager_kwargs if manager_kwargs is not None else {}

        # Default mode (or non-device backend): single task
        if self.mode != 'optimize_session' or not launcher.backend.is_device:

            def run_launcher(launcher=launcher, run_kwargs=run_kwargs) -> Result:
                kwargs = _filter_run_kwargs_for_callable(launcher.run, run_kwargs)
                return launcher.run(**kwargs)

            task = ManagerBackedTask(run_launcher, dependencies=deps, callbacks=cb, pipe_dependencies=False)
            self.tasks.append(task)
            self._classical_tasks.append(task)
            self._manager_kwargs[task] = dict(mkwargs)
            return task

        # optimize_session + device: split into (format) + (quantum run)
        # Formatter depends on classical deps only (to allow quantum block ordering)
        format_deps = [d for d in deps if d not in self._quantum_tasks]
        t_gen = ManagerBackedTask(launcher._get_compatible_problem, dependencies=format_deps, pipe_dependencies=False)
        self._classical_tasks.append(t_gen)
        self._manager_kwargs[t_gen] = dict(mkwargs)

        algo = launcher.algorithm
        backend = launcher.backend

        QLauncherCtor = QLauncher

        def quantum_run(
            formatted_problem: Problem | Model,
            *_: object,
            algo=algo,
            backend=backend,
            run_kwargs=run_kwargs,
            QLauncherCtor=QLauncherCtor,
        ) -> Result:
            ql = QLauncherCtor(formatted_problem, algo, backend)
            kwargs = _filter_run_kwargs_for_callable(ql.run, run_kwargs)
            return ql.run(**kwargs)

        # Quantum run depends on formatter + quantum deps (to serialize device session)
        quantum_deps = [d for d in deps if d in self._quantum_tasks]
        t_quant = ManagerBackedTask(
            quantum_run,
            dependencies=[t_gen] + quantum_deps,
            callbacks=cb,
            pipe_dependencies=True,
        )
        self._quantum_tasks.append(t_quant)
        self._manager_kwargs[t_quant] = dict(mkwargs)

        # User-visible task is the quantum run
        self.tasks.append(t_quant)
        return t_quant

    def start(self) -> None:
        """
        Start scheduling and execution in a background thread.

        This validates that no tasks were previously submitted, prepares session barriers for
        ``optimize_session`` mode (once), and then starts the scheduler loop.

        Args:
                None.

        Returns:
                None.

        Raises:
                ValueError: If the scheduler is already running or tasks were already submitted/finished.
        """
        if self._scheduler_thread is not None and self._scheduler_thread.is_alive():
            raise ValueError('Cannot start again, scheduler is already running.')

        for t in self._classical_tasks + self._quantum_tasks:
            if t.job_id() is not None or t.done() or t.running():
                raise ValueError('Cannot start again, some tasks were already submitted or finished.')

        if self.mode == 'optimize_session' and not self._prepared_optimize_session:
            self._prepare_optimize_session()
            self._prepared_optimize_session = True

        self._scheduler_done.clear()
        self._scheduler_exc = None

        self._scheduler_thread = Thread(target=weakref.proxy(self)._scheduler_loop, daemon=True)
        self._scheduler_thread.start()

    def _prepare_optimize_session(self) -> None:
        """
        Prepare the internal dependency graph for ``optimize_session`` mode.

        This method inserts two "gateway" tasks to enforce the global ordering:
        (classical prerequisites of quantum) -> (all quantum tasks) -> (remaining classical).

        Args:
                None.

        Returns:
                None.
        """
        if not self._quantum_tasks:
            return

        # Collect transitive dependencies of quantum tasks
        quantum_dependencies: set[ManagerBackedTask] = set()
        queue = self._quantum_tasks.copy()
        seen: set[ManagerBackedTask] = set()

        while queue:
            t = queue.pop(0)
            if t in seen:
                continue
            seen.add(t)
            for dep in t.dependencies:
                quantum_dependencies.add(dep)
                queue.append(dep)

        # Classical deps are those deps that are not quantum tasks themselves
        quantum_dependencies = quantum_dependencies.difference(set(self._quantum_tasks))

        # Gateway: after all classical prereqs of quantum tasks
        gateway_classical = ManagerBackedTask(_gateway_true, dependencies=list(quantum_dependencies))
        self._classical_tasks.append(gateway_classical)
        self._manager_kwargs[gateway_classical] = {}

        for qt in self._quantum_tasks:
            qt.dependencies.append(gateway_classical)

        # Gateway: after all quantum tasks
        gateway_quantum = ManagerBackedTask(_gateway_true, dependencies=self._quantum_tasks.copy())
        self._quantum_tasks.append(gateway_quantum)
        self._manager_kwargs[gateway_quantum] = {}

        for ct in [t for t in self._classical_tasks if t not in quantum_dependencies and t is not gateway_classical]:
            ct.dependencies.append(gateway_quantum)

    def _scheduler_loop(self) -> None:
        """
        Run the AQL scheduler loop.

        The scheduler:
        - submits tasks that are ready,
        - waits for any job to finish via the manager,
        - reads results and finalizes tasks,
        - propagates scheduler-level failures to remaining tasks.

        Args:
                None.

        Returns:
                None.
        """
        all_tasks: list[ManagerBackedTask] = []
        for t in self._classical_tasks + self._quantum_tasks:
            if t not in all_tasks:
                all_tasks.append(t)

        job_to_task: dict[str, ManagerBackedTask] = {}

        try:
            while True:
                # Exit condition: all tasks are terminal (DONE/CANCELLED/FAILED)
                if all(t.done() for t in all_tasks):
                    break

                for t in all_tasks:
                    if t.cancelled() or t.done() or t.job_id() is not None:
                        continue
                    if not t.is_ready():
                        continue

                    # IMPORTANT: don't let a single submit exception kill the whole scheduler
                    try:
                        jid = t._submit(self._manager, **self._manager_kwargs.get(t, {}))
                        job_to_task[jid] = t
                    except BaseException as e:
                        t._set_exception(e)

                # Defensive reconciliation: make sure we track all in-flight jobs.
                for t in all_tasks:
                    jid = t.job_id()
                    if jid is None:
                        continue
                    if t.done() or t.cancelled():
                        continue
                    job_to_task.setdefault(jid, t)

                for jid, t in list(job_to_task.items()):
                    if t.done() or t.cancelled() or t.job_id() != jid:
                        job_to_task.pop(jid, None)

                # If no jobs are in-flight and tasks remain -> dependency deadlock
                if not job_to_task:
                    pending = [t for t in all_tasks if (not t.done()) and (not t.cancelled()) and (t.job_id() is None)]
                    if not pending:
                        break
                    raise RuntimeError('Deadlock: no runnable tasks (check dependency graph).')

                try:
                    wait_ret = self._manager.wait_for_a_job(None, timeout=0.5)
                except TimeoutError:
                    continue

                jid = wait_ret[0] if isinstance(wait_ret, tuple) and wait_ret else wait_ret

                if jid is None:
                    continue

                task = job_to_task.pop(jid, None)
                if task is None:
                    task = next((t for t in all_tasks if t.job_id() == jid), None)
                    if task is None:
                        continue

                try:
                    res = self._manager.read_results(jid)
                    task._set_result(res)
                except BaseException as e:
                    task._set_exception(e)

        except BaseException as e:
            # Scheduler-level error: fail remaining tasks so result() doesn't hang
            self._scheduler_exc = e
            for t in all_tasks:
                try:
                    if not t.done() and not t.cancelled():
                        t._set_exception(e)
                except Exception:
                    pass
            with contextlib.suppress(Exception):
                self.cancel_running_tasks()

        finally:
            with contextlib.suppress(Exception):
                self._manager.clean_up()
            self._scheduler_done.set()

    def __enter__(self):
        """
        Enter a context-manager scope.

        Returns:
                The current ``AQL`` instance.
        """
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Exit a context-manager scope and perform best-effort cleanup.

        Args:
                exc_type: Exception type (if any).
                exc_value: Exception instance (if any).
                exc_traceback: Traceback (if any).

        Returns:
                None.
        """
        self.cancel_running_tasks()
        with contextlib.suppress(Exception):
            self._manager.clean_up()
