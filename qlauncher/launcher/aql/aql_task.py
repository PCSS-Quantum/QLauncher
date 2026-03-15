import contextlib
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from threading import Event, Lock
from typing import Any

from qlauncher.base.base import Result
from qlauncher.workflow.base_job_manager import BaseJobManager


def get_timeout(max_timeout: int | float | None, start: int | float) -> float | None:
    """
    Compute remaining time budget when multiple waits share a single overall timeout.

    Args:
            max_timeout: Total allowed timeout in seconds. If None, wait indefinitely.
            start: Start timestamp from time.time().

    Returns:
            Remaining timeout in seconds, or None if max_timeout was None.
    """
    if max_timeout is None:
        return None
    return max_timeout - (time.time() - start)


@dataclass(frozen=True)
class _State:
    PENDING: str = 'PENDING'
    SUBMITTED: str = 'SUBMITTED'
    DONE: str = 'DONE'
    CANCELLED: str = 'CANCELLED'
    FAILED: str = 'FAILED'


class ManagerBackedTask:
    """
    Async-like task that is executed by a BaseJobManager.
    Returned to users. Supports dependencies, callbacks, cancellation, and result() waiting.
    """

    def __init__(
        self,
        task: Callable,
        dependencies: list['ManagerBackedTask'] | None = None,
        callbacks: list[Callable] | None = None,
        pipe_dependencies: bool = False,
    ) -> None:
        """
        Create a task executed by a BaseJobManager.

        Args:
                task: Callable to execute. If pipe_dependencies=True, it will be invoked as task(*dep_results).
                dependencies: Tasks that must be terminal before this task can be submitted.
                callbacks: Optional callables invoked after completion with either (result) or (exception).
                pipe_dependencies: If True, results of dependencies are passed as positional arguments to task.

        Returns:
                None.
        """
        self.task = task
        self.dependencies = dependencies if dependencies is not None else []
        self.callbacks = callbacks if callbacks is not None else []
        self.pipe_dependencies = pipe_dependencies

        self._state = _State.PENDING
        self._state_lock = Lock()

        self._job_id: str | None = None
        self._manager_ref: weakref.ReferenceType[BaseJobManager] | None = None

        self._result: Result | None = None
        self._exception: BaseException | None = None

        self._done_event = Event()

    def is_ready(self) -> bool:
        """
        Check whether the task can be submitted.

        Args:
                None.

        Returns:
                True if all dependency tasks are terminal (DONE/FAILED/CANCELLED), otherwise False.
        """
        return all(dep.done() for dep in self.dependencies)

    def job_id(self) -> str | None:
        return self._job_id

    def cancelled(self) -> bool:
        return self._state == _State.CANCELLED

    def done(self) -> bool:
        return self._state in (_State.DONE, _State.CANCELLED, _State.FAILED)

    def running(self) -> bool:
        return self._state == _State.SUBMITTED and not self._done_event.is_set()

    def cancel(self) -> bool:
        """
        Attempt to cancel the task.

        If the task has already been submitted, this forwards the cancellation to the manager
        best-effort (manager.cancel(job_id)). The task is marked CANCELLED and becomes terminal.

        Args:
                None.

        Returns:
                True if the task is now cancelled (or was already cancelled).
                False if the task was already terminal (DONE/FAILED) and cannot be cancelled.
        """
        with self._state_lock:
            # already terminal
            if self._state in (_State.DONE, _State.FAILED):
                return False
            if self._state == _State.CANCELLED:
                return True

            # pending => cancel locally
            self._state = _State.CANCELLED
            self._done_event.set()

            job_id = self._job_id
            mref = self._manager_ref

        # if submitted, best-effort cancel via manager
        mgr = mref() if mref is not None else None
        if mgr is not None and job_id is not None:
            with contextlib.suppress(Exception):
                mgr.cancel(job_id)

        return True

    def _set_result(self, res: Result) -> None:
        """
        Mark the task as successfully completed and run callbacks.

        Args:
                res: Result value produced by the job.

        Returns:
                None.
        """
        with self._state_lock:
            if self.done():
                return
            self._result = res
            self._state = _State.DONE
            self._done_event.set()

        for cb in self.callbacks:
            with contextlib.suppress(Exception):
                cb(res)

    def _set_exception(self, e: BaseException) -> None:
        """
        Mark the task as failed and run callbacks.

        Args:
                e: Exception raised by the job.

        Returns:
                None.
        """
        with self._state_lock:
            if self.done():
                return
            self._exception = e
            self._state = _State.FAILED
            self._done_event.set()

        for cb in self.callbacks:
            with contextlib.suppress(Exception):
                cb(e)

    def result(self, timeout: float | int | None = None) -> Result | None:
        """
        Wait for the task to reach a terminal state and return its outcome.

        Args:
                timeout: Maximum time to wait in seconds. If None, wait indefinitely.

        Returns:
                The task result if completed successfully.
                None if the task was cancelled.

        Raises:
                TimeoutError: If timeout expires before the task becomes terminal (task is cancelled).
                BaseException: Re-raises the exception produced by the job when the task is FAILED.
        """
        start = time.time()
        if not self._done_event.wait(timeout=get_timeout(timeout, start)):
            self.cancel()
            raise TimeoutError

        if self._state == _State.CANCELLED:
            return None
        if self._exception is not None:
            raise self._exception
        return self._result

    def _submit(self, manager: BaseJobManager, **manager_kwargs) -> str:
        """
        Submit this task to a job manager and transition it to SUBMITTED.

        This method is intended to be called by a scheduler (e.g., AQL) once is_ready() is True.
        If pipe_dependencies=True, dependency results are collected and passed as positional args.

        Args:
                manager: Job manager used to execute the task.
                **manager_kwargs: Manager-specific arguments forwarded to manager.submit(...).

        Returns:
                The job ID returned by manager.submit(...).

        Raises:
                RuntimeError: If the task is not in PENDING state, or if it is already cancelled.
        """
        with self._state_lock:
            if self._state != _State.PENDING:
                raise RuntimeError('Task already submitted or terminal.')
            if self.cancelled():
                raise RuntimeError('Cannot submit a cancelled task.')
            self._manager_ref = weakref.ref(manager)

        task_fn = self.task

        # Evaluate dep results only here (scheduler should call _submit only when is_ready()).
        dep_results: list[Any] = []
        if self.pipe_dependencies:
            dep_results = [d.result() for d in self.dependencies]

        func: Callable[..., Any]
        if self.pipe_dependencies:
            dep_args = tuple(dep_results)

            @wraps(task_fn)
            def bound(task_fn=task_fn, dep_args=dep_args) -> task_fn:
                return task_fn(*dep_args)

            func = bound
        else:
            func = task_fn

        jid = manager.submit(func, **manager_kwargs)
        with self._state_lock:
            self._job_id = jid
            self._state = _State.SUBMITTED
        return jid
