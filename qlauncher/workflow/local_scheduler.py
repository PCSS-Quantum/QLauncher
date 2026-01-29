import time
import weakref
from collections.abc import Callable
from threading import Event, Thread
from typing import Any

import multiprocess
from pathos.multiprocessing import _ProcessPool

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


class _InnerMPTask:
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
		callbacks: list[Callable] | None = None,
	) -> None:
		self.task = task
		self.callbacks = callbacks if callbacks is not None else []

		self._cancelled = False
		self._thread = None
		self._pool = _ProcessPool(processes=1)
		self._thread_made = Event()

		self._result = None
		self._done = False

	def _shutdown_subprocess(self) -> None:
		self._pool.close()
		self._pool.terminate()
		self._pool.join()

	def _async_task(self) -> Any:
		if self._cancelled:
			self._result = None
			self._done = True
			return

		res = self._pool.apply_async(self.task)
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


# Why like this? The inner task was not getting properly garbage collected when it was running,
# but this does and just cancels the inner task so it also gets garbage collected
# this is cursed :/
class MPTask:
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
		callbacks: list[Callable] | None = None,
	) -> None:
		self._inner_task = _InnerMPTask(task, callbacks)
		weakref.finalize(self, self._inner_task.cancel)

	def __getattr__(self, name: str) -> Any:
		return getattr(self._inner_task, name)


class LocalJobManager(BaseJobManager):
	def __init__(self):
		super().__init__()
		self.tasks: dict[str, MPTask] = {}
		self._poll_interval_s: float = 0.05

	def submit(
		self,
		function: Callable,
		**kwargs,
	) -> str:
		jid = self._make_job_uid()
		while jid in self.tasks or jid in self.jobs:
			jid = self._make_job_uid()

		self.jobs[jid] = {'finished': False}

		t = MPTask(function)
		self.tasks[jid] = t
		t.start()
		return jid

	def wait_for_a_job(
		self,
		job_id: str | None = None,
		timeout: float | None = None,
	) -> str | None:
		# czekaj na konkretny job
		if job_id is not None:
			if job_id not in self.tasks:
				raise KeyError('No such job!')
			self.tasks[job_id].result(timeout=timeout)
			if job_id in self.jobs:
				self.jobs[job_id]['finished'] = True
			return job_id

		# czekaj na jakikolwiek job (job_id=None)
		if not self.tasks:
			raise ValueError('No jobs to wait for.')

		start = time.time()
		while True:
			for jid, t in self.tasks.items():
				# zakończony i jeszcze nieoznaczony jako finished
				if t.done() and (jid in self.jobs) and (not self.jobs[jid].get('finished', False)):
					self.jobs[jid]['finished'] = True
					return jid

			if timeout is not None and (time.time() - start) >= timeout:
				raise TimeoutError

			time.sleep(self._poll_interval_s)

	def read_results(self, job_id: str) -> Result:
		if job_id not in self.tasks:
			raise KeyError('No such job!')
		res = self.tasks[job_id].result()
		if job_id in self.jobs:
			self.jobs[job_id]['finished'] = True
		return res

	def cancel(self, job_id: str):
		if job_id not in self.tasks:
			raise KeyError('No such job!')
		ok = self.tasks[job_id].cancel()
		if ok and job_id in self.jobs:
			self.jobs[job_id]['finished'] = True
		return ok

	def clean_up(self) -> None:
		for t in self.tasks.values():
			t.cancel()
