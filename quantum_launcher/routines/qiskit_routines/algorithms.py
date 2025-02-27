""" Algorithms for Qiskit routines """
import json
from datetime import datetime
import math
import os

import numpy as np
from qiskit import qpy, QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
# from qiskit.opflow import H
from qiskit.primitives.base.base_primitive import BasePrimitive
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.minimum_eigensolvers import QAOA as QiskitQAOA
from qiskit_algorithms.minimum_eigensolvers import SamplingVQEResult
import scipy

from quantum_launcher.base import Problem, Algorithm, Result
from .backend import QiskitBackend
from quantum_launcher.workflow.pilotjob_scheduler import JobManager
from typing import Callable


class QiskitOptimizationAlgorithm(Algorithm):
    """ Abstract class for Qiskit optimization algorithms """

    def make_tag(self, problem: Problem, backend: QiskitBackend) -> str:
        tag = problem.__class__.__name__ + '-' + \
            backend.__class__.__name__ + '-' + \
            self.__class__.__name__ + '-' + \
            datetime.today().strftime('%Y-%m-%d')
        return tag

    def get_processing_times(self, tag: str, primitive: BasePrimitive) -> None | tuple[list, list, int]:
        timestamps = []
        usages = []
        qpu_time = 0
        if hasattr(primitive, 'session'):
            jobs = primitive.session.service.jobs(limit=None, job_tags=[tag])
            for job in jobs:
                m = job.metrics()
                timestamps.append(m['timestamps'])
                usages.append(m['usage'])
                qpu_time += m['usage']['quantum_seconds']
        return timestamps, usages, qpu_time


def commutator(op_a: SparsePauliOp, op_b: SparsePauliOp) -> SparsePauliOp:
    """ Commutator """
    return op_a @ op_b - op_b @ op_a


class QAOA(QiskitOptimizationAlgorithm):
    """Algorithm class with QAOA.

    Args:
        p (int): The number of QAOA steps. Defaults to 1.
        alternating_ansatz (bool): Whether to use an alternating ansatz. Defaults to False. If True, it's recommended to provide a mixer_h to alg_kwargs.
        aux: Auxiliary input for the QAOA algorithm.
        **alg_kwargs: Additional keyword arguments for the base class.

    Attributes:
        name (str): The name of the algorithm.
        aux: Auxiliary input for the QAOA algorithm.
        p (int): The number of QAOA steps.
        alternating_ansatz (bool): Whether to use an alternating ansatz.
        parameters (list): List of parameters for the algorithm.
        mixer_h (SparsePauliOp | None): The mixer Hamiltonian.
        mixer_h (QuantumCircuit | None): The initial state of the circuit.

    """
    _algorithm_format = 'hamiltonian'

    def __init__(self, p: int = 1, alternating_ansatz: bool = False, aux=None, **alg_kwargs):
        super().__init__(**alg_kwargs)
        self.name: str = 'qaoa'
        self.aux = aux
        self.p: int = p
        self.alternating_ansatz: bool = alternating_ansatz
        self.parameters = ['p']
        self.mixer_h: SparsePauliOp | None = None
        self.initial_state: QuantumCircuit | None = None

    @property
    def setup(self) -> dict:
        return {
            'aux': self.aux,
            'p': self.p,
            'parameters': self.parameters,
            'arg_kwargs': self.alg_kwargs
        }

    def parse_samplingVQEResult(self, res: SamplingVQEResult, res_path) -> dict:
        res_dict = {}
        for k, v in vars(res).items():
            if k[0] == "_":
                key = k[1:]
            else:
                key = k
            try:
                res_dict = {**res_dict, **json.loads(json.dumps({key: v}))}
            except TypeError as ex:
                if str(ex) == 'Object of type complex128 is not JSON serializable':
                    res_dict = {**res_dict, **
                                json.loads(json.dumps({key: v}, default=repr))}
                elif str(ex) == 'Object of type ndarray is not JSON serializable':
                    res_dict = {**res_dict, **
                                json.loads(json.dumps({key: v}, default=repr))}
                elif str(ex) == 'keys must be str, int, float, bool or None, not ParameterVectorElement':
                    res_dict = {**res_dict, **
                                json.loads(json.dumps({key: repr(v)}))}
                elif str(ex) == 'Object of type OptimizerResult is not JSON serializable':
                    # recursion ftw
                    new_v = self.parse_samplingVQEResult(v, res_path)
                    res_dict = {**res_dict, **
                                json.loads(json.dumps({key: new_v}))}
                elif str(ex) == 'Object of type QuantumCircuit is not JSON serializable':
                    path = res_path + '.qpy'
                    with open(path, 'wb') as f:
                        qpy.dump(v, f)
                    res_dict = {**res_dict, **{key: path}}
        return res_dict

    def run(self, problem: Problem, backend: QiskitBackend, formatter:Callable = None) -> Result:
        """ Runs the QAOA algorithm """
        hamiltonian: SparsePauliOp = formatter(problem)
        energies = []

        def qaoa_callback(evaluation_count, params, mean, std):
            energies.append(mean)

        tag = self.make_tag(problem, backend)
        sampler = backend.samplerV1
        # sampler.set_options(job_tags=[tag])
        optimizer = backend.optimizer

        if self.alternating_ansatz:
            if self.mixer_h is None:
                self.mixer_h = formatter.get_mixer_hamiltonian(problem)
            if self.initial_state is None:
                self.initial_state = formatter.get_QAOAAnsatz_initial_state(
                    problem)

        qaoa = QiskitQAOA(sampler, optimizer, reps=self.p, callback=qaoa_callback,
                          mixer=self.mixer_h, initial_state=self.initial_state, **self.alg_kwargs)
        qaoa_result = qaoa.compute_minimum_eigenvalue(hamiltonian, self.aux)
        depth = qaoa.ansatz.decompose(reps=10).depth()
        if 'cx' in qaoa.ansatz.decompose(reps=10).count_ops():
            cx_count = qaoa.ansatz.decompose(reps=10).count_ops()['cx']
        else:
            cx_count = 0
        timestamps, usages, qpu_time = self.get_processing_times(tag, sampler)
        return self.construct_result({'energy': qaoa_result.eigenvalue,
                                      'depth': depth,
                                      'cx_count': cx_count,
                                      'qpu_time': qpu_time,
                                      'energies': energies,
                                      'SamplingVQEResult': qaoa_result,
                                      'usages': usages,
                                      'timestamps': timestamps})

    def construct_result(self, result: dict) -> Result:

        best_bitstring = self.get_bitstring(result)
        best_energy = result['energy']

        distribution = dict(result['SamplingVQEResult'].eigenstate.items())
        most_common_value = max(
            distribution, key=distribution.get)
        most_common_bitstring = bin(most_common_value)[2:].zfill(
            len(best_bitstring))
        most_common_bitstring_energy = distribution[most_common_value]
        num_of_samples = 0  # TODO: implement
        average_energy = np.mean(result['energies'])
        energy_std = np.std(result['energies'])
        return Result(best_bitstring, best_energy, most_common_bitstring, most_common_bitstring_energy, distribution, result['energies'], num_of_samples, average_energy, energy_std, result)

    def get_bitstring(self, result) -> str:
        return result['SamplingVQEResult'].best_measurement['bitstring']


class FALQON(QiskitOptimizationAlgorithm):
    """ 
    Algorithm class with FALQON.

    Args:
        driver_h (Optional[Operator]): The driver Hamiltonian for the problem.
        delta_t (float): The time step for the evolution operators.
        beta_0 (float): The initial value of beta.
        n (int): The number of iterations to run the algorithm.
        **alg_kwargs: Additional keyword arguments for the base class.

    Attributes:
        driver_h (Optional[Operator]): The driver Hamiltonian for the problem.
        delta_t (float): The time step for the evolution operators.
        beta_0 (float): The initial value of beta.
        n (int): The number of iterations to run the algorithm.
        cost_h (Optional[Operator]): The cost Hamiltonian for the problem.
        n_qubits (int): The number of qubits in the problem.
        parameters (List[str]): The list of algorithm parameters.

    """

    def __init__(self, driver_h=None, delta_t=0, beta_0=0, n=1):
        super().__init__()
        self.driver_h = driver_h
        self.delta_t = delta_t
        self.beta_0 = beta_0
        self.n = n
        self.cost_h = None
        self.n_qubits: int = 0
        self.parameters = ['n', 'delta_t', 'beta_0']
        raise NotImplementedError('FALQON is not implemented yet')

    @property
    def setup(self) -> dict:
        return {
            'driver_h': self.driver_h,
            'delta_t': self.delta_t,
            'beta_0': self.beta_0,
            'n': self.n,
            'cost_h': self.cost_h,
            'n_qubits': self.n_qubits,
            'parameters': self.parameters,
            'arg_kwargs': self.alg_kwargs
        }

    def _get_path(self) -> str:
        return f'{self.name}@{self.n}@{self.delta_t}@{self.beta_0}'

    def run(self, problem: Problem, backend: QiskitBackend):
        """ Runs the FALQON algorithm """
        # TODO implement aux operator
        hamiltonian = problem.get_qiskit_hamiltonian()
        self.cost_h = hamiltonian
        self.n_qubits = hamiltonian.num_qubits
        if self.driver_h is None:
            self.driver_h = SparsePauliOp.from_sparse_list(
                [("X", [i], 1) for i in range(self.n_qubits)], num_qubits=self.n_qubits)

        betas = [self.beta_0]
        energies = []
        circuit_depths = []
        cxs = []

        tag = self.make_tag(problem, backend)
        estimator = backend.estimator
        sampler = backend.sampler
        sampler.set_options(job_tags=[tag])
        estimator.set_options(job_tags=[tag])

        best_sample, last_sample = self._falqon_subroutine(estimator,
                                                           sampler, energies, betas, circuit_depths, cxs)

        timestamps, usages, qpu_time = self.get_processing_times(tag, sampler)
        result = {'betas': betas,
                  'energies': energies,
                  'depths': circuit_depths,
                  'cxs': cxs,
                  'n': self.n,
                  'delta_t': self.delta_t,
                  'beta_0': self.beta_0,
                  'energy': min(energies),
                  'qpu_time': qpu_time,
                  'best_sample': best_sample,
                  'last_sample': last_sample,
                  'usages': usages,
                  'timestamps': timestamps}

        return result

    def _build_ansatz(self, betas):
        """ building ansatz circuit """
        H = None  # TODO: implement H
        circ = (H ^ self.cost_h.num_qubits).to_circuit()
        params = ParameterVector("beta", length=len(betas))
        for param in params:
            circ.append(PauliEvolutionGate(
                self.cost_h, time=self.delta_t), circ.qubits)
            circ.append(PauliEvolutionGate(self.driver_h,
                        time=self.delta_t * param), circ.qubits)
        return circ

    def _falqon_subroutine(self, estimator,
                           sampler, energies, betas, circuit_depths, cxs):
        """ subroutine for falqon """
        for i in range(self.n):
            betas, energy, depth, cx_count = self._run_falqon(betas, estimator)
            print(i, energy)
            energies.append(energy)
            circuit_depths.append(depth)
            cxs.append(cx_count)
        argmin = np.argmin(np.asarray(energies))
        best_sample = self._sample_at(betas[:argmin], sampler)
        last_sample = self._sample_at(betas, sampler)
        return best_sample, last_sample

    def _run_falqon(self, betas, estimator):
        """ Method to run FALQON algorithm """
        ansatz = self._build_ansatz(betas)
        comm_h = complex(0, 1) * commutator(self.driver_h, self.cost_h)
        beta = -1 * estimator.run(ansatz, comm_h, betas).result().values[0]
        betas.append(beta)

        ansatz = self._build_ansatz(betas)
        energy = estimator.run(ansatz, self.cost_h, betas).result().values[0]

        depth = ansatz.decompose(reps=10).depth()
        if 'cx' in ansatz.decompose(reps=10).count_ops():
            cx_count = ansatz.decompose(reps=10).count_ops()['cx']
        else:
            cx_count = 0

        return betas, energy, depth, cx_count

    def _sample_at(self, betas, sampler):
        """ Not sure yet """
        ansatz = self._build_ansatz(betas)
        ansatz.measure_all()
        res = sampler.run(ansatz, betas).result()
        return res


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

    def run(self, problem: Problem, backend: QiskitBackend, formatter) -> Result:
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
