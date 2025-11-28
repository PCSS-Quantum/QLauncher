import contextlib
import os
import pickle
import shutil
import subprocess
import sys
import time
from typing import Any

from qlauncher.base.base import Algorithm, Backend, Problem, Result
from qlauncher.exceptions import DependencyError

try:
	import dill
except ImportError as e:
	raise DependencyError(e, install_hint='dill') from e


class SlurmJobManager:
	def __init__(
		self,
		sbatch_exe: str = 'sbatch',
		slurm_options: dict[str, Any] | None = None,
		env_setup: list[str] | None = None,
	) -> None:
		self.jobs: dict[str, dict[str, Any]] = {}
		self.code_path = os.path.join(os.path.dirname(__file__), 'pilotjob_task.py')

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
		problem: Problem,
		algorithm: Algorithm,
		backend: Backend,
		cores: int = 1,
	) -> str:
		from qlauncher.launcher.qlauncher import QLauncher

		launcher = QLauncher(problem, algorithm, backend)
		return self.submit_launcher(launcher, cores=cores)

	def submit_launcher(self, launcher, cores: int = 1) -> str:
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

	def read_results(self, job_id: str) -> Result:
		job = self.jobs[job_id]
		output_file = job['output_file']
		if not os.path.exists(output_file):
			raise FileNotFoundError(f'Result file for job {job_id} not found: {output_file}')
		with open(output_file, 'rb') as rt:
			result: Result = pickle.load(rt)
		job['finished'] = True
		return result

	def wait_for_a_job(self, job_id: str | None = None, timeout: float | None = None) -> str | None:
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
				if os.path.exists(output_file):
					return job_id
				raise RuntimeError(f'Job {job_id} disappeared from squeue but result file does not exist: {output_file}')

			if state in ('PENDING', 'CONFIGURING', 'RUNNING', 'COMPLETING'):
				time.sleep(2.0)
				continue

			if state in ('COMPLETED', 'COMPLETING', 'CG'):
				if not os.path.exists(output_file):
					raise RuntimeError(f'Job {job_id} finished with state {state}, but result file not found: {output_file}')
				return job_id

			raise RuntimeError(f'Job {job_id} finished in bad state: {state}')

	def clean_up(self) -> None:
		for job in self.jobs.values():
			for key in ('script_path', 'input_file'):
				path = job.get(key)
				if path and os.path.exists(path):
					with contextlib.suppress(OSError):
						os.remove(path)

	def __del__(self) -> None:
		with contextlib.suppress(Exception):
			self.clean_up()

	def _make_job_uid(self) -> str:
		return f'{len(self.jobs):05d}'

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

	def _get_slurm_state(self, job_id: str) -> str | None:
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
