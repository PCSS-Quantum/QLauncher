import time
import weakref
from collections.abc import Callable

from multiprocess.context import TimeoutError as MPTimeoutError
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


def _shutdown_subprocess(pool) -> None:
	pool.close()
	pool.terminate()
	pool.join()


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
	) -> None:
		self.task = task

		self._cancelled = False

		self._pool = _ProcessPool(processes=1)
		self._pool_process = None

		weakref.finalize(self, _shutdown_subprocess, self._pool)

	def start(self) -> None:
		"""Start task execution."""
		if self._pool_process is not None or self._cancelled:
			raise ValueError('Cannot start, task already started or cancelled.')
		self._pool_process = self._pool.apply_async(self.task)

	def cancel(self) -> bool:
		"""
		Attempt to cancel the task.

		Returns:
			bool: True if cancellation was successful
		"""
		_shutdown_subprocess(self._pool)
		self._cancelled = True
		self._pool_process = None
		return True

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
		return self._cancelled or self._pool_process.ready()

	def running(self) -> bool:
		"""
		Returns:
			bool: True if the task is currently executing.
		"""
		return self._pool_process is not None and not self._cancelled

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
		try:
			if self._cancelled:
				return None
			return self._pool_process.get(timeout=timeout) if self._pool_process is not None else None
		except MPTimeoutError as e:
			raise TimeoutError from e


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
