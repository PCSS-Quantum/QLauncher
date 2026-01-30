"""Wrapper for QLauncher that enables the user to launch tasks asynchronously (futures + multiprocessing)"""

import time
import weakref
from collections.abc import Callable
from threading import Event, Thread
from typing import Any

from qlauncher.base.base import Result
from qlauncher.workflow.base_job_manager import BaseJobManager


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


class _IAQLTask:
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
		manager: BaseJobManager,
		dependencies: list['AQLTask'] | None = None,
		callbacks: list[Callable] | None = None,
		pipe_dependencies: bool = False,
	) -> None:
		self.task = task
		self.dependencies = dependencies if dependencies is not None else []
		self.callbacks = callbacks if callbacks is not None else []
		self.pipe_dependencies = pipe_dependencies

		self._cancelled = False
		self._thread = None
		self._manager = manager
		self._thread_made = Event()

		self._result = None
		self._done = False

	def _async_task(self) -> Any:
		dep_results = [d.result() for d in self.dependencies]

		if self._cancelled:
			self._result = None
			self._done = True
			return

		task = self.task
		pipe = self.pipe_dependencies

		def run():
			return task(*dep_results) if pipe else task()

		jid = self._manager.submit(run)
		while not self._cancelled:
			try:
				self._manager.wait_for_a_job(jid, timeout=0.05)
				self._result = self._manager.read_results(jid)
				self._done = True
				return
			except TimeoutError:
				pass  # task not ready, check for cancel
			# For any other error originating from the task, shutdown and clean up subprocess then raise error again.
			except Exception as e:
				self._manager.cancel(jid)
				self._result = e
				self._done = True
				return

		if self._cancelled:
			self._manager.cancel(jid)  # kill res process
			self._result = None

		self._done = True
		return

	def _target_task(self) -> None:
		# Main task + callbacks launch
		self._async_task()
		for cb in self.callbacks:
			cb(self._result)

	def _set_thread(self) -> None:
		self._thread = Thread(target=weakref.proxy(self)._target_task, daemon=True)  # set daemon so that thread quits as main process quits
		self._thread.start()
		self._thread_made.set()

	def start(self) -> None:
		"""Start task execution."""
		if self._thread is not None or self._cancelled:
			raise ValueError('Cannot start, task already started or cancelled.')
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


class AQLTask:
	def __init__(
		self,
		task: Callable,
		manager: BaseJobManager,
		dependencies: list['AQLTask'] | None = None,
		callbacks: list[Callable] | None = None,
		pipe_dependencies: bool = False,
	) -> None:
		self._inner_task = _IAQLTask(task, manager, dependencies, callbacks, pipe_dependencies)
		weakref.finalize(self, self._inner_task.cancel)

	def __getattr__(self, name: str) -> Any:
		return getattr(self._inner_task, name)
