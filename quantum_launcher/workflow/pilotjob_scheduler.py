import os
import pickle
import sys
from typing import List, Optional

from quantum_launcher.base.base import Algorithm, Backend, Problem, Result
from quantum_launcher.launcher.qlauncher import QuantumLauncher
from quantum_launcher.exceptions import DependencyError
try:
    import dill
    from qcg.pilotjob.api.job import Jobs
    from qcg.pilotjob.api.manager import LocalManager, Manager
except ImportError as e:
    raise DependencyError(e, install_hint='pilotjob') from e


class JobManager:
    def __init__(self, manager: Optional[Manager] = None):
        """
        Job manager is Quantum Launcher's wrapper for process management system, current version works on top of qcg-pilotjob

        Args:
            manager (Optional[Manager], optional): Manager system to schedule jobs, if set to None, the pilotjob's LocalManager is set.
            Defaults to None.
        """
        self.jobs = {}
        self.code_path = os.path.join(os.path.dirname(__file__), 'pilotjob_task.py')
        if manager is None:
            manager = LocalManager()
        self.manager = manager

    def _count_not_finished(self) -> int:
        return len([job for job in self.jobs.values() if job.get('finished') is False])

    def submit(self, problem: Problem, algorithm: Algorithm, backend: Backend, output_path: str, cores: Optional[int] = None) -> str:
        """
        Submits Quantum Launcher job to the scheduler

        Args:
            problem (Problem): Problem.
            algorithm (Algorithm): Algorithm.
            backend (Backend): Backend.
            output_path (str): Path of output file.
            cores (Optional[int], optional): Number of cores per task, if None value set to number of free cores (at least 1). Defaults to None.

        Returns:
            str: Job Id.
        """
        if cores is None:
            cores = max(1, self.manager.resources()['free_cores'])
        job = self._prepare_ql_dill_job(problem=problem, algorithm=algorithm, backend=backend,
                                        output=output_path, cores=cores)
        job_id = self.manager.submit(Jobs().add(**job.get('qcg_args')))[0]
        return job_id

    def submit_many(self, problem: Problem, algorithm: Algorithm, backend: Backend, output_path: str, cores_per_job: int = 1) -> List[str]:
        """
        Submits as many jobs as there are currently available cores.

        Args:
            problem (Problem): Problem.
            algorithm (Algorithm): Algorithm.
            backend (Backend): Backend.
            output_path (str): Path of output file.
            cores_per_job (int, optional): Number of cores per job. Defaults to 1.

        Returns:
            List[str]: List with Job Id's.
        """
        free_cores = self.manager.resources()['free_cores']
        if free_cores == 0:
            return []
        qcg_jobs = Jobs()
        for _ in range(free_cores//cores_per_job):
            job = self._prepare_ql_dill_job(problem=problem, algorithm=algorithm, backend=backend, output=output_path, cores=cores_per_job)
            qcg_jobs.add(**job.get('qcg_args'))
        return self.manager.submit(qcg_jobs)

    def wait_for_a_job(self, job_id: Optional[str] = None, timeout: Optional[int | float] = None) -> tuple[str, str]:
        """
        Waits for a job to finish and returns it's id and status.

        Args:
            job_id (Optional[str], optional): Id of selected job, if None waiting for any job. Defaults to None.
            timeout (Optional[int  |  float], optional): Timeout in seconds. Defaults to None.

        Raises:
            ValueError: Raises if job_id not found or there are no jobs left.

        Returns:
            tuple[str, str]: job_id, job's status
        """
        if job_id is None:
            if self._count_not_finished() <= 0:
                raise ValueError("There are no jobs left")
            job_id, state = self.manager.wait4_any_job_finish(timeout)
        elif job_id in self.jobs:
            state = self.manager.wait4(job_id, timeout=timeout)[job_id]
        else:
            raise ValueError(f"Job {job_id} not found in {self.__class__.__name__}'s jobs")

        self.jobs[job_id]['finished'] = True
        return job_id, state

    def _prepare_ql_dill_job(self, problem: Problem, algorithm: Algorithm, backend: Backend,
                             output: str, cores: int = 1) -> dict:
        job_uid = f'{len(self.jobs):05d}'
        output_file = os.path.abspath(f'{output}output.{job_uid}')
        launcher = QuantumLauncher(problem, algorithm, backend)
        input_file = os.path.abspath(f'{output}output.{job_uid}')
        with open(input_file, 'wb') as f:
            dill.dump(launcher, f)

        in_args = [self.code_path, input_file, output_file]
        qcg_args = {
            'name': job_uid,
            'exec': sys.executable,
            'args': in_args,
            'model': 'openmpi',
            'stdout': f'{output}stdout.{job_uid}',
            'stderr': f'{output}stderr.{job_uid}',
            'numCores': cores
        }
        job = {'name': job_uid, 'qcg_args': qcg_args, 'output_file': output_file, 'finished': False}
        self.jobs[job_uid] = job
        return job

    def read_results(self, job_id: str) -> Result:
        """
        Reads the result of given job_id.

        Args:
            job_id (str): Job Id.

        Returns:
            Result: Result of selected Job.
        """
        output_path = self.jobs[job_id]['output_file']
        with open(output_path, 'rb') as rt:
            results = pickle.load(rt)
        return results

    def clean_up(self):
        """
        Removes all output files generated in the process and calls self.manager.cleanup().
        """
        for job in self.jobs.values():
            os.remove(job['output_file'])
        if isinstance(self.manager, LocalManager):
            self.manager.cleanup()

    def stop(self):
        """
        Stops the manager process.
        """
        self.manager.cancel(self.jobs)
        self.manager.finish()

    def __del__(self):
        self.stop()
