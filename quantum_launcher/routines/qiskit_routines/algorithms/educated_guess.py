""" Algorithms for Qiskit routines """
import math
import os

import numpy as np
import scipy

from quantum_launcher.base import Problem, Algorithm, Result
from quantum_launcher.routines.qiskit_routines.backends.ibm_backend import IBMBackend
from quantum_launcher.workflow.pilotjob_scheduler import JobManager
from quantum_launcher.routines.qiskit_routines.algorithms.qiskit_native import QAOA


class EducatedGuess(Algorithm):
    _algorithm_format = 'hamiltonian'

    def __init__(self, starting_p: int = 3, max_p: int = 8, verbose: bool = False):
        """
        Algorithm utilizing all available cores to run multiple QAOA's in parallel to find optimal parameters.

        Args:
            starting_p (int, optional): Initial value of QAOA's p parameter. Defaults to 3.
            max_p (int, optional): Maximum value for QAOA's p parameter. Defaults to 8.
            verbose (bool, optional): Verbose. Defaults to False.
        """
        self.output_initial = 'initial/'
        self.output_interpolated = 'interpolated/'
        self.output = 'output/'
        self.p_init = starting_p
        self.p_max = max_p
        self.verbose = verbose
        self.failed_jobs = 0
        self.min_energy = math.inf
        self.manager = JobManager()
        self.best_job_id = ''

    def run(self, problem: Problem, backend: IBMBackend, formatter) -> Result:
        self.manager.submit_many(problem, QAOA(p=self.p_init), backend, output_path=self.output_initial)
        print(f'{len(self.manager.jobs)} jobs submitted to qcg')

        found_optimal_params = False

        while not found_optimal_params:
            jobid, state = self.manager.wait_for_a_job()

            if state != 'SUCCEED':
                self.failed_jobs += 1
                continue
            has_potential, energy = self._process_job(jobid, self.p_init, self.min_energy, compare_factor=0.99)
            if has_potential:
                found_optimal_params = self._search_for_job_with_optimal_params(jobid, energy, problem, backend)

            self.manager.submit_many(problem, QAOA(p=self.p_init), backend, output_path=self.output_initial)

        result = self.manager.read_results(self.best_job_id)
        self.manager.stop()
        return result

    def _search_for_job_with_optimal_params(self, previous_job_id, previous_energy, problem, backend) -> bool:
        for p in range(self.p_init + 1, self.p_max + 1):
            previous_job_results = self.manager.read_results(previous_job_id).result
            initial_point = self._interpolate_f(list(previous_job_results['SamplingVQEResult'].optimal_point), p-1)

            new_job_id = self.manager.submit(problem, QAOA(p=p, initial_point=initial_point),
                                             backend, output_path=self.output_interpolated)
            _, state = self.manager.wait_for_a_job(new_job_id)
            if state != 'SUCCEED':
                self.failed_jobs += 1
                return False
            has_potential, new_energy = self._process_job(new_job_id, p, previous_energy)
            if has_potential:
                previous_energy = new_energy
                previous_job_id = new_job_id
            else:
                return False
        self.best_job_id = new_job_id
        return True

    def _process_job(self, jobid: str, p: int, energy_to_compare: float, compare_factor: float = 1.0) -> tuple[
            float, bool]:
        result = self.manager.read_results(jobid).result
        optimal_point = result['SamplingVQEResult'].optimal_point
        has_potential = False
        linear = self._check_linearity(optimal_point, p)
        energy = result['energy']
        if self.verbose:
            print(f'job {jobid}, p={p}, energy: {energy}')

        if p == self.p_init and energy < energy_to_compare:
            print(f'new min energy: {energy}')
            self.min_energy = energy
        if linear and energy * compare_factor < energy_to_compare:
            has_potential = True
        return has_potential, energy

    def _create_directories_if_not_existing(self):
        if not os.path.exists(self.output_initial):
            os.makedirs(self.output_initial)
        if not os.path.exists(self.output_interpolated):
            os.makedirs(self.output_interpolated)
        if not os.path.exists(self.output):
            os.makedirs(self.output)

    def _interp(self, params: np.ndarray) -> np.ndarray:
        arr1 = np.append([0], params)
        arr2 = np.append(params, [0])
        weights = np.arange(len(arr1)) / len(params)
        res = arr1 * weights + arr2 * weights[::-1]
        return res

    def _interpolate_f(self, params: np.ndarray, p: int) -> np.ndarray:
        betas = params[:p]
        gammas = params[p:]
        new_betas = self._interp(betas)
        new_gammas = self._interp(gammas)
        return np.hstack([new_betas, new_gammas])

    def _check_linearity(self, optimal_params: np.ndarray, p: int) -> bool:
        linear = False
        correlations = (scipy.stats.pearsonr(np.arange(1, p + 1), optimal_params[:p])[0],
                        scipy.stats.pearsonr(np.arange(1, p + 1), optimal_params[p:])[0])

        if abs(correlations[0]) > 0.85 and abs(correlations[1]) > 0.85:
            linear = True
        return linear
