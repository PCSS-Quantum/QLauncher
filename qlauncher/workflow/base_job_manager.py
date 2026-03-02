from abc import ABC, abstractmethod
from typing import Any

from qlauncher.base import Algorithm, Backend, Problem, Model, Result


class BaseJobManager(ABC):
	"""
	Abstract base class for job managers that execute QLauncher jobs
	on different compute backends.
	"""

	def __init__(self):
		self.jobs: dict[str, dict[str, Any]] = {}

	@abstractmethod
	def submit(
		self,
		problem: Problem | Model,
		algorithm: Algorithm,
		backend: Backend,
		cores: int = 1,
		**kwargs,
	) -> str:
		"""
		Submit a QLauncher job to the scheduler.

		Args:
			problem: Problem to be solved.
			algorithm: Algorithm to be used.
			backend: Backend on which the algorithm will be executed.
			cores: Number of CPU cores per task.
			**kwargs: Manager-specific additional arguments.

		Returns:
			Job ID as a string.
		"""
		pass

	@abstractmethod
	def wait_for_a_job(
		self,
		job_id: str | None = None,
		timeout: float | None = None,
	) -> str | None:
		"""
		Wait for a job to finish and return its ID.

		Args:
			job_id: ID of the job to wait for. If None, wait for any job.
			timeout: Maximum time to wait in seconds. If None, wait indefinitely.

		Returns:
			Job ID of the finished job.

		Raises:
			ValueError: If no jobs are available to wait for.
			TimeoutError: If timeout is exceeded.
		"""
		pass

	@abstractmethod
	def read_results(self, job_id: str) -> Result:
		"""
		Read the result of a finished job.

		Args:
			job_id: Job ID returned by submit().

		Returns:
			Result object produced by the job.

		Raises:
			KeyError: If job_id is not known to this manager.
			FileNotFoundError: If the result file does not exist.
		"""
		pass

	@abstractmethod
	def clean_up(self) -> None:
		"""
		Clean up temporary files and resources created by the manager.
		"""
		pass

	def run(
		self,
		problem: Problem | Model,
		algorithm: Algorithm,
		backend: Backend,
		cores: int = 1,
		**kwargs,
	) -> Result:
		"""
		Convenience method: submit job, wait for completion, read results, and cleanup.

		This method handles the complete lifecycle of a job execution.

		Args:
			problem: Problem to be solved.
			algorithm: Algorithm to be used.
			backend: Backend on which the algorithm will be executed.
			cores: Number of CPU cores per task.
			**kwargs: Manager-specific additional arguments.

		Returns:
			Result object produced by the job.
		"""
		try:
			job_id = self.submit(problem, algorithm, backend, cores=cores, **kwargs)
			self.wait_for_a_job(job_id)
			return self.read_results(job_id)
		finally:
			self.clean_up()

	def _count_not_finished(self) -> int:
		"""Count how many jobs are not yet marked as finished."""
		return len([job for job in self.jobs.values() if not job.get('finished', False)])

	def _make_job_uid(self) -> str:
		"""Generate a unique job identifier."""
		return f'{len(self.jobs):05d}'


if __name__ == '__main__':
	from qlauncher.problems import MaxCut
	from qlauncher.routines.qiskit import QAOA, QiskitBackend

	problem = MaxCut.from_preset('default')
	algorithm = QAOA(p=3)
	backend = QiskitBackend('local_simulator')

	# SlurmJobManager
	from qlauncher.workflow.slurm_job_manager import SlurmJobManager

	slurm_mgr = SlurmJobManager(
		slurm_options={
			'time': '00:30:00',
			# "licenses": "orca1:1",
		},
		env_setup=[
			# "module load Python/python-3.11.0",
			# "source ~/venv/bin/activate",
		],
	)
	slurm_result = slurm_mgr.run(problem, algorithm, backend, cores=1)
	print(slurm_result)

	# # PilotJobManager
	# from qlauncher.workflow.pilotjob_scheduler import PilotJobManager
	#
	# out_dir = Path("./pilotjob_out")
	#
	# pilot_mgr = PilotJobManager()
	# pilot_result = pilot_mgr.run(
	# 	problem,
	# 	algorithm,
	# 	backend,
	# 	cores=1,
	# 	output_path=out_dir,
	# )
	# print(pilot_result)
