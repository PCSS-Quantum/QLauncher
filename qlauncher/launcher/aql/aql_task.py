"""Wrapper for QLauncher that enables the user to launch tasks asynchronously (futures + multiprocessing)"""
from collections.abc import Callable
from typing import Any
from threading import Event, Thread
import time
import weakref
import multiprocess

from pathos.multiprocessing import _ProcessPool

from qlauncher.base.base import Result


def get_timeout(max_timeout: int | float | None, start: int | float) -> float | None:
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


class _InnerAQLTask:
    """
        Task object returned to user, so that dependencies can be created.

        Attributes:
            task (Callable): function that gets executed asynchronously
            dependencies (list[AQLTask]): Optional dependencies. The task will wait for all its dependencies to finish, before starting.
            callbacks (list[Callable]): Callbacks ran when the task finishes executing. 
                                        Task result is inserted as an argument to the function.
            pipe_dependencies (bool): If True results of tasks defined as dependencies will be passed as arguments to self.task. 
                                      Defaults to False.
    """

    def __init__(
        self,
        task: Callable,
        dependencies: list['AQLTask'] | None = None,
        callbacks: list[Callable] | None = None,
        pipe_dependencies: bool = False
    ) -> None:

        self.task = task
        self.dependencies = dependencies if dependencies is not None else []
        self.callbacks = callbacks if callbacks is not None else []
        self.pipe_dependencies = pipe_dependencies

        self._cancelled = False
        self._thread = None
        self._pool = _ProcessPool(processes=1)
        self._thread_made = Event()

        self._result = None
        self._done = False

    def _shutdown_subprocess(self):
        self._pool.close()
        self._pool.terminate()
        self._pool.join()

    def _async_task(self) -> Any:
        dep_results = [d.result() for d in self.dependencies]

        if self._cancelled:
            self._result = None
            self._done = True
            return

        res = self._pool.apply_async(self.task, args=(dep_results if self.pipe_dependencies else []))
        # Turns out you can't just outright kill threads (or futures) is so I have to do this, so that the thread knows to exit.
        while not self._cancelled:
            try:
                self._result = res.get(timeout=0.05)
                self._done = True
                return
            except multiprocess.context.TimeoutError:
                pass  # task not ready, check for cancel
            # For any other error originating from the task, shutdown and clean up subprocess then raise error again.
            except Exception as e:
                self._shutdown_subprocess()
                self._result = e
                self._done = True
                return

        if self._cancelled:
            self._shutdown_subprocess()  # kill res process

        self._result = res.get() if res.ready() else None
        self._done = True
        return

    def _target_task(self):
        # Main task + callbacks launch
        self._async_task()
        for cb in self.callbacks:
            cb(self._result)

    def _set_thread(self):
        self._thread = Thread(target=weakref.proxy(self)._target_task, daemon=True)  # set daemon so that thread quits as main process quits
        self._thread.start()
        self._thread_made.set()

    def start(self):
        """Start task execution."""
        if self._thread is not None or self._cancelled:
            raise ValueError("Cannot start, task already started or cancelled.")
        self._set_thread()

    def cancel(self) -> bool:
        """
        Attempt to cancel the task.

        Returns:
            bool: True if cancellation was successful
        """
        self._cancelled = True
        if self._thread is None:
            return True
        self._thread.join(0.1)
        return not self._thread.is_alive()

    def cancelled(self) -> bool:
        """
        Returns:
            bool: True if the task was cancelled by the user.
        """
        return self._cancelled

    def done(self) -> bool:
        """
        Returns:
            bool: True if the task had finished execution.
        """
        return self._done

    def running(self) -> bool:
        """
        Returns:
            bool: True if the task is currently executing.
        """
        if self._thread is None:
            return False
        return self._thread.is_alive()

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
        self._thread_made.wait(timeout=get_timeout(timeout, start))  # Wait until we start a thread
        self._thread.join(timeout=get_timeout(timeout, start))
        if self._thread.is_alive():
            self.cancel()
            raise TimeoutError  # thread still running after timeout
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


# Why like this? The inner task was not getting properly garbage collected when it was running,
# but this does and just cancels the inner task so it also gets garbage collected
# this is cursed :/
class AQLTask:
    """
        Task object returned to user, so that dependencies can be created.

        Attributes:
            task (Callable): function that gets executed asynchronously
            dependencies (list[AQLTask]): Optional dependencies. The task will wait for all its dependencies to finish, before starting.
            callbacks (list[Callable]): Callbacks ran when the task finishes executing. 
                                        Task result is inserted as an argument to the function.
            pipe_dependencies (bool): If True results of tasks defined as dependencies will be passed as arguments to self.task. 
                                      Defaults to False.
    """

    def __init__(
        self,
        task: Callable,
        dependencies: list['AQLTask'] | None = None,
        callbacks: list[Callable] | None = None,
        pipe_dependencies: bool = False
    ) -> None:
        self._inner_task = _InnerAQLTask(task, dependencies, callbacks, pipe_dependencies)
        weakref.finalize(self, self._inner_task.cancel)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner_task, name)
