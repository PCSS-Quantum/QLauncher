import contextlib
import time
import weakref
from collections.abc import Callable
from threading import Event, Thread
from typing import TYPE_CHECKING, Any

import multiprocess as mp
from multiprocess.queues import Queue as QueueType

if TYPE_CHECKING:
    from multiprocess.process import BaseProcess as ProcessType
else:
    ProcessType = Any

# LocalJobManager used to run each job via pathos._ProcessPool (apply_async + polling).
# On Windows this proved flaky for our test/CI contract (cancel + GC cleanup): worker pools were not
# deterministically torn down, leaving child processes alive and causing intermittent timeouts and
# noisy Pool.__del__/WinError shutdown errors. I switched to a per-job multiprocess.Process + Queue
# so we can reliably terminate/join the worker on cancel/cleanup and ensure no subprocesses leak.

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


def _run_in_subprocess(q: QueueType, fn: Callable[[], Any]) -> None:
	"""
	Execute a callable in a worker process and send its outcome back via a queue.

	Args:
		q: Queue used to report ("ok", result) or ("err", exception).
		fn: Zero-argument callable to execute in the child process.

	Returns:
		None. The result/exception is returned through the queue.
	"""
	try:
		q.put(("ok", fn()))
	except BaseException as e:
		# Pass exception object back; parent will raise it.
		q.put(("err", e))


class _InnerMPTask:
	"""
	Internal asynchronous task runner implemented as:
	- a supervisor thread in the parent process,
	- a dedicated child process running the user callable,
	- a queue for returning result or exception.

	Args:
		task: Zero-argument callable to execute.
		callbacks: Optional list of callables invoked with the task outcome (result or exception).

	Attributes:
		task: Callable executed in the child process.
		callbacks: Functions called after task finishes (best-effort).
	"""
	def __init__(
		self,
		task: Callable,
		callbacks: list[Callable] | None = None,
	) -> None:
		self.task = task
		self.callbacks = callbacks if callbacks is not None else []

		self._cancelled = False
		self._thread: Thread | None = None
		self._thread_made = Event()

		self._proc: ProcessType | None = None
		self._queue: QueueType | None = None

		self._result: Any = None
		self._done = False

	def _terminate_proc(self) -> None:
		p = self._proc
		if p is None:
			return
		try:
			if p.is_alive():
				p.terminate()
		except Exception:
			pass
		with contextlib.suppress(Exception):
			p.join(timeout=1.0)

	def _async_task(self) -> None:
		if self._cancelled:
			self._result = None
			self._done = True
			return

		ctx = mp.get_context()  # na Windows będzie spawn (domyślnie)
		q: QueueType = ctx.Queue(maxsize=1)
		self._queue = q

		p = ctx.Process(target=_run_in_subprocess, args=(q, self.task), daemon=True)
		self._proc = p

		try:
			p.start()
		except BaseException as e:
			self._result = e
			self._done = True
			return

		while True:
			if self._cancelled:
				self._terminate_proc()
				self._result = None
				self._done = True
				return

			try:
				p.join(timeout=0.05)
			except Exception:
				time.sleep(0.05)

			if not p.is_alive():
				break

		try:
			tag, payload = q.get_nowait()
		except Exception:
			tag, payload = ("ok", None)

		if tag == "err":
			self._result = payload
		else:
			self._result = payload

		self._done = True

	def thread_main(self) -> None:
		self._async_task()

		for cb in self.callbacks:
			with contextlib.suppress(Exception):
				cb(self._result)

	def _set_thread(self) -> None:
		self._thread = Thread(target=weakref.proxy(self).thread_main, daemon=True)  # set daemon so that thread quits as main process quits
		self._thread.start()
		self._thread_made.set()

	def start(self) -> None:
		"""
		Start task execution.

		Args:
			None.

		Returns:
			None.
		"""
		if self._thread is not None or self._cancelled:
			raise ValueError('Cannot start, task already started or cancelled.')
		self._set_thread()

	def cancel(self) -> bool:
		"""
		Attempt to cancel the task.

		Args:
			None.

		Returns:
			True if the task is considered canceled.
		"""
		self._cancelled = True
		if self._thread is None:
			self._terminate_proc()
			self._result = None
			self._done = True
			return True

		# Started: terminate subprocess and wait briefly for supervisor thread to exit
		self._terminate_proc()
		with contextlib.suppress(Exception):
			self._thread.join(timeout=1.0)

		# Mark as done (even if thread is stubborn); public contract: cancelled => terminal
		self._result = None
		self._done = True
		return True

	def cancelled(self) -> bool:
		"""
		Returns:
			bool: True if the task was canceled by the user.
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
		Wait for the task to finish and return its result.

		Args:
			timeout: Maximum time to wait in seconds. If None, wait indefinitely.

		Returns:
			The task result, or None if the task was canceled.

		Raises:
			TimeoutError: If the task does not finish within the timeout (task is canceled).
			BaseException: Re-raises an exception produced by the task.
		"""
		start = time.time()

		# wait until the supervisor thread exists
		self._thread_made.wait(timeout=get_timeout(timeout, start))

		# if never started, just return current result
		if self._thread is None:
			return None

		self._thread.join(timeout=get_timeout(timeout, start))
		if self._thread.is_alive():
			self.cancel()
			raise TimeoutError  # thread still running after timeout
		if isinstance(self._result, BaseException):
			raise self._result
		return self._result


class MPTask:
	"""
	Public-facing wrapper around _InnerMPTask.

	This wrapper exists so that if the task object is garbage-collected,
	we best-effort cancel the underlying execution to avoid leaking subprocesses.

	Args:
		task: Zero-argument callable to execute.
		callbacks: Optional list of callables invoked with the task outcome.

	Attributes:
		_inner_task: The underlying _InnerMPTask instance.
	"""

	def __init__(
		self,
		task: Callable,
		callbacks: list[Callable] | None = None,
	) -> None:
		self._inner_task = _InnerMPTask(task, callbacks)
		weakref.finalize(self, self._inner_task.cancel)

	def __getattr__(self, name: str):
		return getattr(self._inner_task, name)


class LocalJobManager(BaseJobManager):
	"""
	Run jobs locally using a per-job child process.

	This manager implements the BaseJobManager interface and is primarily intended
	for local execution and testing (supports waiting for any job, cancellation, and cleanup).
	"""
	def __init__(self, poll_interval_s: float = 0.05):
		super().__init__()
		self.tasks: dict[str, MPTask] = {}
		self._poll_interval_s = poll_interval_s

	def __del__(self):
		with contextlib.suppress(Exception):
			self.clean_up()

	def submit(
		self,
		function: Callable,
		**kwargs,
	) -> str:
		"""
		Submit a function job to the scheduler.

		Args:
			function: Function to be executed.
			**kwargs: Manager-specific additional arguments (currently unused by LocalJobManager).

		Returns:
			Job ID as a string.
		"""
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
		"""
		Wait for a job to finish.

		Args:
			job_id: If provided, wait for this specific job. If None, wait for any unfinished job.
			timeout: Maximum time to wait in seconds. If None, wait indefinitely.

		Returns:
			The finished job ID.

		Raises:
			KeyError: If job_id is provided but unknown.
			ValueError: If waiting for any job but no jobs exist.
			TimeoutError: If timeout expires before a job finishes.
		"""
		if job_id is not None:
			if job_id not in self.tasks:
				raise KeyError('No such job!')
			self.tasks[job_id].result(timeout=timeout)
			if job_id in self.jobs:
				self.jobs[job_id]['finished'] = True
			return job_id

		if not self.tasks:
			raise ValueError('No jobs to wait for.')

		start = time.time()
		while True:
			for jid, t in self.tasks.items():
				if jid in self.jobs and self.jobs[jid].get("finished", False):
					continue
				if t.done() or t.cancelled():
					if jid in self.jobs:
						self.jobs[jid]["finished"] = True
					return jid

			if timeout is not None and (time.time() - start) >= timeout:
				raise TimeoutError

			time.sleep(self._poll_interval_s)

	def read_results(self, job_id: str) -> Result:
		"""
		Read results for a finished job.

		Args:
			job_id: Job ID.

		Returns:
			The job result.

		Raises:
			KeyError: If job_id is unknown.
			BaseException: Re-raises an exception produced by the job.
		"""
		if job_id not in self.tasks:
			raise KeyError('No such job!')
		res = self.tasks[job_id].result()
		if job_id in self.jobs:
			self.jobs[job_id]['finished'] = True
		return res

	def cancel(self, job_id: str) -> bool:
		"""
		Cancel a submitted job.

		Args:
			job_id: Job ID.

		Returns:
			True if cancellation was performed (best-effort) and the job is marked finished.

		Raises:
			KeyError: If job_id is unknown.
		"""
		if job_id not in self.tasks:
			raise KeyError('No such job!')
		ok = self.tasks[job_id].cancel()
		if ok and job_id in self.jobs:
			self.jobs[job_id]['finished'] = True
		return ok

	def clean_up(self) -> None:
		"""
		Cancel all known jobs and mark them as finished.

		Args:
			None.

		Returns:
			None.
		"""
		for jid, t in list(self.tasks.items()):
			with contextlib.suppress(Exception):
				t.cancel()
			if jid in self.jobs:
				self.jobs[jid]["finished"] = True
