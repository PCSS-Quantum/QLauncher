import json
import os
import pickle
import sys
from typing import Any, Dict, List, Optional
from qcg.pilotjob.api.errors import TimeoutElapsed
from qcg.pilotjob.api.job import Jobs
from qcg.pilotjob.api.manager import LocalManager
from quantum_launcher.base.base import Algorithm, Backend, Problem, Result


class JobManager:
    def __init__(self, manager_args: Optional[List[str]] = None):
        if manager_args is None:
            manager_args = []
        self.jobs = {}
        self.code_path = os.path.join(os.path.dirname(__file__), 'pilotjob_task.py')
        self.manager = LocalManager(manager_args)

    def not_finished(self):
        return len([job for job in self.jobs.values() if job.get('finished') is False])

    def submit(self, problem, algorithm, backend, output_path: str, kwargs: Dict[str, Any]):
        free_cores = self.manager.resources()['free_cores']
        number_of_cores = max(1, free_cores)
        job = self.prepare_ql_job(problem=problem, algorithm=algorithm, backend=backend,
                                  output=output_path, cores=number_of_cores, kwargs=kwargs)
        job_id = self.manager.submit(Jobs().add(**job.get('qcg_args')))[0]
        return job_id

    def submit_many(self, problem, algorithm, backend, output_path: str, kwargs: Optional[Dict[str, Any]] = None):
        if kwargs is None:
            kwargs = {}
        free_cores = self.manager.resources()['free_cores']
        if free_cores == 0:
            return
        qcg_jobs = Jobs()
        for _ in range(free_cores):
            job = self.prepare_ql_job(problem=problem, algorithm=algorithm, backend=backend, output=output_path, kwargs=kwargs)
            qcg_jobs.add(**job.get('qcg_args'))
        return self.manager.submit(qcg_jobs)

    def wait_for_a_job(self, job_id: Optional[str] = None) -> tuple[str, str] | None:
        while self.not_finished() > 0:
            try:
                if job_id is None:
                    job_id, state = self.manager.wait4_any_job_finish(10)
                else:
                    state = self.manager.wait4(job_id)[job_id]
                if job_id not in self.jobs:
                    print(f'error: job {job_id} not known', flush=True)
                    continue

                self.jobs[job_id]['finished'] = True
                return job_id, state

            except TimeoutElapsed:
                continue
            except Exception as ex:
                print(f'error in waiting: {str(ex)}', flush=True)
                continue

    def prepare_ql_job(self, problem: Problem, algorithm: Algorithm, backend: Backend, output: str, cores: int = 1, kwargs=None) -> dict:
        if kwargs is None:
            kwargs = {}
        job_uid = f'{len(self.jobs):05d}'
        output_file = os.path.abspath(f'{output}output.{job_uid}')
        kwargs_str = json.dumps(kwargs)
        in_args = [self.code_path, problem.__class__.__name__, algorithm.__class__.__name__,
                   backend.__class__.__name__, output_file, kwargs_str]
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

    def read_results(self, job_id) -> Result:
        output_path = self.jobs[job_id]['output_file']
        with open(output_path, 'rb') as rt:
            results = pickle.load(rt)
        return results

    def __del__(self):
        self.manager.cancel(self.jobs)
        self.manager.kill_manager_process()
