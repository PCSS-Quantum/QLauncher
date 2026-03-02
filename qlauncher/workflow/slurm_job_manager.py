import contextlib
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from qlauncher.base import Algorithm, Backend, Problem, Model, Result
from qlauncher.exceptions import DependencyError
from qlauncher.launcher import QLauncher
from qlauncher.workflow.base_job_manager import BaseJobManager

try:
	import dill
except ImportError as e:
	raise DependencyError(e, install_hint='pilotjob') from e


class SlurmJobManager(BaseJobManager):
	def __init__(
		self,
		sbatch_exe: str = 'sbatch',
		slurm_options: dict[str, Any] | None = None,
		env_setup: list[str] | None = None,
	) -> None:
		"""
		Job manager that submits QLauncher jobs to Slurm via ``sbatch``.

		Args:
			sbatch_exe (str, optional): Name or path of the ``sbatch`` executable
				used to submit jobs to Slurm. Defaults to ``"sbatch"``.
			slurm_options (dict[str, Any] | None, optional): Mapping of Slurm
				options to their values (e.g. ``{"time": "00:02:00"}``).
				Keys are used as option names after ``--`` in the generated
				``#SBATCH`` lines. Defaults to an empty dict.
			env_setup (list[str] | None, optional): List of shell commands that
				will be written into the Slurm script before the ``srun`` line,
				e.g. module loads or virtual environment activation commands.
				Defaults to an empty list.

		Raises:
			DependencyError: If the ``sbatch_exe`` executable cannot be found
				in ``PATH``.
		"""
		super().__init__()
		self.code_path = Path(__file__).with_name('pilotjob_task.py')
		self.sbatch_exe = sbatch_exe
		self.slurm_options = slurm_options or {}
		self.env_setup = env_setup or []

		if shutil.which(self.sbatch_exe) is None:
			raise DependencyError(
				ImportError(f'{self.sbatch_exe} not found in PATH'),
				install_hint='slurm',
			)

	def submit(
		self,
		problem,
		algorithm,
		backend,
		cores: int = 1,
		**kwargs,
	) -> str:
		"""
		Creates a :class:`QLauncher`
		instance from ``problem``, ``algorithm`` and ``backend`` and forwards
		it to :meth:`submit_launcher`.

		Args:
			problem (Problem | Model): Problem to be solved.
			algorithm (Algorithm): Algorithm to be used.
			backend (Backend): Backend on which the algorithm will be executed.
			cores (int, optional): Number of CPU cores per task requested from
				Slurm (mapped to ``--cpus-per-task``). Defaults to 1.

		Returns:
			str: Slurm job ID returned by ``sbatch``.

		Raises:
			RuntimeError: If ``sbatch`` returns a non-zero exit code.
		"""
		launcher = QLauncher(problem, algorithm, backend)
		return self.submit_launcher(launcher, cores=cores)

	def submit_launcher(self, launcher, cores: int = 1):
		"""
		Submits a prepared :class:`QLauncher` instance to Slurm.

		Args:
			launcher (QLauncher): Prepared launcher object.
			cores (int, optional): Number of CPU cores per task requested from
				Slurm (mapped to ``--cpus-per-task``). Defaults to 1.

		Returns:
			str: Slurm job ID returned by ``sbatch``.

		Raises:
			RuntimeError: If ``sbatch`` returns a non-zero exit code or its
				standard output does not contain a job ID.
		"""
		job_uid = self._make_job_uid()

		input_file = f'input.{job_uid}.pkl'
		output_file = f'output.{job_uid}.pkl'
		script_path = f'slurm_job.{job_uid}.sh'

		with open(input_file, 'wb') as f:
			dill.dump(launcher, f)

		self._write_sbatch_script(
			script_path=script_path,
			job_uid=job_uid,
			input_file=input_file,
			output_file=output_file,
			cores=cores,
		)

		res = subprocess.run(
			[self.sbatch_exe, script_path],
			capture_output=True,
			text=True,
			check=False,
		)

		if res.returncode != 0:
			raise RuntimeError(f'sbatch failed ({res.returncode}): {res.stderr}')

		job_id = res.stdout.strip().split()[-1]

		self.jobs[job_id] = {
			'uid': job_uid,
			'input_file': input_file,
			'output_file': output_file,
			'script_path': script_path,
			'finished': False,
		}
		return job_id

	def wait_for_a_job(
		self,
		job_id: str | None = None,
		timeout: float | None = None,
	):
		"""
		Waits until a Slurm job finishes and returns its ID.

		Args:
			job_id (str | None, optional): ID of the job to wait for. If
				``None``, the first job in :attr:`jobs` that is not yet marked
				as finished is selected. Defaults to ``None``.
			timeout (float | None, optional): Maximum time to wait in seconds.
				If ``None``, wait indefinitely. Defaults to ``None``.

		Raises:
			ValueError: If ``job_id`` is ``None`` and there are no jobs left.
			TimeoutError: If the timeout is exceeded before the job finishes.
			RuntimeError: If the job disappears from ``squeue`` without
				producing a result file, or if it finishes in a non-successful
				state.

		Returns:
			str: ID of the finished job.
		"""
		if job_id is None:
			not_finished = [jid for jid, j in self.jobs.items() if not j['finished']]
			if not not_finished:
				raise ValueError('There are no jobs left')
			job_id = not_finished[0]

		job = self.jobs[job_id]
		output_file = job['output_file']

		start = time.time()

		while True:
			now = time.time()
			if timeout is not None and (now - start) > timeout:
				raise TimeoutError(f'Timeout waiting for job {job_id}')

			state = self._get_slurm_state(job_id)

			if state is None:
				if Path(output_file).exists():
					job['finished'] = True
					return job_id
				raise RuntimeError(f'Job {job_id} disappeared from squeue but result file does not exist: {output_file}')

			if state in ('PENDING', 'CONFIGURING', 'RUNNING', 'COMPLETING'):
				time.sleep(2.0)
				continue

			if state in ('COMPLETED', 'CG'):
				if not Path(output_file).exists():
					raise RuntimeError(f'Job {job_id} finished with state {state}, but result file not found: {output_file}')
				job['finished'] = True
				return job_id

			raise RuntimeError(f'Job {job_id} finished in bad state: {state}')

	def read_results(self, job_id):
		"""
		Reads the result of a finished job from its output file.

		Args:
			job_id (str): Slurm job ID returned by :meth:`submit` or
				:meth:`submit_launcher`.

		Raises:
			KeyError: If ``job_id`` is not known to this manager.
			FileNotFoundError: If the expected output file does not exist.

		Returns:
			Result: Deserialized result object produced by the worker process.
		"""
		if job_id not in self.jobs:
			raise KeyError(f'Job {job_id} not found')

		job = self.jobs[job_id]
		output_file = job['output_file']

		if not Path(output_file).exists():
			raise FileNotFoundError(f'Result file for job {job_id} not found: {output_file}')

		with open(output_file, 'rb') as rt:
			result: Result = pickle.load(rt)

		job['finished'] = True
		return result

	def clean_up(self):
		"""
		Removes temporary files created for all tracked jobs.
		"""
		for job in self.jobs.values():
			for key in ('script_path', 'input_file'):
				path = job.get(key)
				if path and Path(path).exists():
					with contextlib.suppress(OSError):
						os.remove(path)

	def _write_sbatch_script(
		self,
		script_path: str,
		job_uid: str,
		input_file: str,
		output_file: str,
		cores: int,
	) -> None:
		opts = dict(self.slurm_options)
		opts.setdefault('ntasks', 1)
		opts.setdefault('cpus-per-task', cores)

		stdout_file = f'stdout.{job_uid}'
		stderr_file = f'stderr.{job_uid}'

		with open(script_path, 'w', encoding='utf-8') as sh:
			sh.write('#!/bin/bash\n')
			sh.write(f'#SBATCH --job-name=ql_{job_uid}\n')
			sh.write(f'#SBATCH --output={stdout_file}\n')
			sh.write(f'#SBATCH --error={stderr_file}\n')

			for opt, val in opts.items():
				sh.write(f'#SBATCH --{opt}={val}\n')

			for line in self.env_setup:
				sh.write(line + '\n')

			sh.write(f'srun {sys.executable} {self.code_path} {input_file} {output_file}\n')

	@staticmethod
	def _get_slurm_state(job_id: str) -> str | None:
		try:
			res = subprocess.run(
				['squeue', '-h', '-j', job_id, '-o', '%T'],
				capture_output=True,
				text=True,
				check=False,
			)
		except FileNotFoundError:
			return None

		if res.returncode != 0:
			return None

		out = res.stdout.strip()
		if not out:
			return None

		return out.splitlines()[0].strip()
