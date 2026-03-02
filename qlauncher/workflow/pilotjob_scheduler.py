import os
import pickle
import sys
from pathlib import Path
import contextlib
from typing import Any


from qlauncher.base import Algorithm, Backend, Problem, Model, Result
from qlauncher.exceptions import DependencyError
from qlauncher.workflow.base_job_manager import BaseJobManager

try:
	import dill
	from qcg.pilotjob.api.job import Jobs
	from qcg.pilotjob.api.manager import LocalManager, Manager
except ImportError as e:
	raise DependencyError(e, install_hint='pilotjob') from e


class PilotJobManager(BaseJobManager):
	def __init__(self, manager: Manager | None = None):
		"""
		PilotJob manager is QLauncher's wrapper for process management system, current version works on top of qcg-pilotjob

		Args:
			manager (Manager | None, optional): Manager system to schedule jobs, if set to None, the pilotjob's LocalManager is set.
			Defaults to None.
		"""
		super().__init__()
		self.code_path = os.path.join(os.path.dirname(__file__), 'pilotjob_task.py')
		self.manager = manager if manager is not None else LocalManager()

	def submit(
		self,
		problem: Problem | Model,
		algorithm,
		backend,
		cores: int = 1,
		output_path: str | None = None,
		**kwargs,
	) -> str:
		"""
		Submits QLauncher job to the scheduler

		Args:
			problem (Problem): Problem.
			algorithm (Algorithm): Algorithm.
			backend (Backend): Backend.
			output_path (str): Path of output file.
			cores (int | None, optional): Number of cores per task, if None value set to number of free cores (at least 1). Defaults to None.

		Returns:
			str: Job Id.
		"""
		if output_path is None:
			raise ValueError('output_path is required for PilotJobManager')

		job = self._prepare_ql_dill_job(
			problem=problem,
			algorithm=algorithm,
			backend=backend,
			output=output_path,
			cores=cores,
		)
		return self.manager.submit(Jobs().add(**job['qcg_args']))[0]

	def submit_many(
		self,
		problem: Problem | Model,
		algorithm,
		backend,
		output_path,
		cores_per_job: int = 1,
		n_jobs: int | None = None,
	) -> list[str]:
		"""
		Submits as many jobs as there are currently available cores.

		Args:
			problem (Problem): Problem.
			algorithm (Algorithm): Algorithm.
			backend (Backend): Backend.
			output_path (str): Path of output file.
			cores_per_job (int, optional): Number of cores per job. Defaults to 1.
			n_jobs: number of jobs to submit. If None, submit as many as possible (free_cores//cores_per_job). Defaults to None.

		Returns:
			list[str]: List with Job Id's.
		"""
		free_cores = self.manager.resources()['free_cores']
		if free_cores == 0:
			return []

		qcg_jobs = Jobs()
		num_jobs = n_jobs if n_jobs is not None else free_cores // cores_per_job

		for _ in range(num_jobs):
			job = self._prepare_ql_dill_job(
				problem=problem,
				algorithm=algorithm,
				backend=backend,
				output=output_path,
				cores=cores_per_job,
			)
			qcg_jobs.add(**job['qcg_args'])

		return self.manager.submit(qcg_jobs)

	def wait_for_a_job(
		self,
		job_id: str | None = None,
		timeout: float | None = None,
	) -> tuple[str, str]:
		"""
		Waits for a job to finish and returns it's id and status.

		Args:
			job_id (str | None, optional): Id of selected job, if None waiting for any job. Defaults to None.
			timeout (int  |  float | None, optional): Timeout in seconds. Defaults to None.

		Raises:
			ValueError: Raises if job_id not found or there are no jobs left.

		Returns:
			tuple[str, str]: job_id, job's status
		"""
		if job_id is None:
			if self._count_not_finished() <= 0:
				raise ValueError('There are no jobs left')
			job_id, state = self.manager.wait4_any_job_finish(timeout)
		elif job_id in self.jobs:
			state = self.manager.wait4(job_id, timeout=timeout)[job_id]
		else:
			raise ValueError(f"Job {job_id} not found in {self.__class__.__name__}'s jobs")

		self.jobs[job_id]['finished'] = True
		return job_id, state

	def _prepare_ql_dill_job(
		self,
		problem: Model,
		algorithm: Algorithm,
		backend: Backend,
		output: str,
		cores: int = 1,
	) -> dict:
		from qlauncher.launcher.qlauncher import QLauncher

		job_uid = self._make_job_uid()

		out_dir = Path(output).expanduser().resolve()
		out_dir.mkdir(parents=True, exist_ok=True)

		output_file = out_dir / f'output.{job_uid}.pkl'
		input_file = out_dir / f'input.{job_uid}.pkl'
		stdout_file = out_dir / f'stdout.{job_uid}'
		stderr_file = out_dir / f'stderr.{job_uid}'

		launcher = QLauncher(problem, algorithm, backend)

		with open(input_file, 'wb') as f:
			dill.dump(launcher, f)

		in_args = [self.code_path, str(input_file), str(output_file)]
		qcg_args = {
			'name': job_uid,
			'exec': sys.executable,
			'args': in_args,
			'model': 'openmpi',
			'stdout': str(stdout_file),
			'stderr': str(stderr_file),
			'numCores': cores,
		}

		job = {
			'name': job_uid,
			'qcg_args': qcg_args,
			'output_file': str(output_file),
			'finished': False,
		}
		self.jobs[job_uid] = job
		return job

	def read_results(self, job_id) -> Any:
		"""
		Reads the result of given job_id.

		Args:
			job_id (str): Job Id.

		Returns:
			Result: Result of selected Job.
		"""
		if job_id not in self.jobs:
			raise KeyError(f'Job {job_id} not found')

		output_path = self.jobs[job_id]['output_file']
		with open(output_path, 'rb') as rt:
			return pickle.load(rt)

	def cancel(self, job_id: str) -> None:
		"""
		Cancel a given job

		Args:
			job_id (str): id of the job to cancel.

		Raises:
			KeyError: If job with a given id was not submitted by this manager.

		Returns:
			None
		"""
		if job_id not in self.jobs:
			raise KeyError(f'Job {job_id} not found')
		return self.manager.cancel(job_id)

	def clean_up(self) -> None:
		"""
		Removes all output files generated in the process and calls self.manager.cleanup().
		"""
		for job in self.jobs.values():
			if os.path.exists(job['output_file']):
				os.remove(job['output_file'])

		if isinstance(self.manager, LocalManager):
			with contextlib.suppress(Exception):
				self.manager.cleanup()

	def stop(self) -> None:
		"""
		Stops the manager process.
		"""
		mgr = getattr(self, 'manager', None)
		if mgr is None:
			return

		if isinstance(mgr, LocalManager):
			with contextlib.suppress(Exception):
				mgr.finish()

	def __del__(self):
		with contextlib.suppress(Exception):
			self.stop()
