""" Algorithms for Qiskit routines """
import json
from datetime import datetime
from collections.abc import Callable

import numpy as np

from qiskit import qpy, QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate

from qiskit.primitives import PrimitiveResult, SamplerPubResult, BaseEstimatorV1, BaseSamplerV1
from qiskit.primitives.containers import BitArray
from qiskit.primitives.base.base_primitive import BasePrimitive

from qiskit.quantum_info import SparsePauliOp

from qiskit_algorithms.minimum_eigensolvers import QAOA as QiskitQAOA
from qiskit_algorithms.minimum_eigensolvers import SamplingVQEResult
from qiskit_algorithms.optimizers import Optimizer, COBYLA

from qlauncher.base import Problem, Algorithm, Result
from qlauncher.base.base import Backend
from qlauncher.routines.cirq_routines import CirqBackend
from qlauncher.routines.qiskit_routines.backends.qiskit_backend import QiskitBackend


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
        optimizer (Optimizer | None): Optimizer used during algorithm runtime. If set to `None` turns into COBYLA. Defaults to None,
        alternating_ansatz (bool): Whether to use an alternating ansatz. Defaults to False. If True, it's recommended to provide a mixer_h to alg_kwargs.
        aux: Auxiliary input for the QAOA algorithm.
        **alg_kwargs: Additional keyword arguments for the base class.

    Attributes:
        name (str): The name of the algorithm.
        aux: Auxiliary input for the QAOA algorithm.
        p (int): The number of QAOA steps.
        optimizer (Optimizer): Optimizer used during algorithm runtime.
        alternating_ansatz (bool): Whether to use an alternating ansatz.
        parameters (list): List of parameters for the algorithm.
        mixer_h (SparsePauliOp | None): The mixer Hamiltonian.

    """
    _algorithm_format = 'hamiltonian'

    def __init__(self, p: int = 1, optimizer: Optimizer | None = None, alternating_ansatz: bool = False, aux=None, **alg_kwargs):
        super().__init__(**alg_kwargs)
        self.name: str = 'qaoa'
        self.aux = aux
        self.p: int = p
        self.optimizer: Optimizer = optimizer if optimizer is not None else COBYLA()
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

    def run(self, problem: Problem, backend: Backend, formatter: Callable) -> Result:
        """ Runs the QAOA algorithm """
        if not (isinstance(backend, QiskitBackend) or isinstance(backend, CirqBackend)):
            raise ValueError('Backend should be CirqBackend, QiskitBackend or subclass.')
        hamiltonian: SparsePauliOp = formatter(problem)
        energies = []

        def qaoa_callback(evaluation_count, params, mean, std):
            energies.append(mean)

        tag = self.make_tag(problem, backend)
        sampler = backend.samplerV1
        # sampler.set_options(job_tags=[tag])

        if self.alternating_ansatz:
            if self.mixer_h is None:
                self.mixer_h = formatter.get_mixer_hamiltonian(problem)
            if self.initial_state is None:
                self.initial_state = formatter.get_QAOAAnsatz_initial_state(
                    problem)

        qaoa = QiskitQAOA(sampler, self.optimizer, reps=self.p, callback=qaoa_callback,
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
    _algorithm_format = 'hamiltonian'

    def __init__(
        self,
        driver_h: SparsePauliOp | None = None,
        delta_t: float = 0.03,
        beta_0: float = 0.0,
        max_reps: int = 20
    ) -> None:
        super().__init__()
        self.driver_h = driver_h
        self.cost_h = None
        self.delta_t = delta_t
        self.beta_0 = beta_0
        self.max_reps = max_reps
        self.n_qubits: int = 0
        self.parameters = ['n', 'delta_t', 'beta_0']

    @property
    def setup(self) -> dict:
        return {
            'driver_h': self.driver_h,
            'delta_t': self.delta_t,
            'beta_0': self.beta_0,
            'n': self.max_reps,
            'cost_h': self.cost_h,
            'n_qubits': self.n_qubits,
            'parameters': self.parameters,
            'arg_kwargs': self.alg_kwargs
        }

    def _get_path(self) -> str:
        return f'{self.name}@{self.max_reps}@{self.delta_t}@{self.beta_0}'

    def run(self, problem: Problem, backend: QiskitBackend, formatter: Callable) -> Result:
        """ Runs the FALQON algorithm """

        if isinstance(backend.sampler, BaseSamplerV1) or isinstance(backend.estimator, BaseEstimatorV1):
            raise ValueError("FALQON works only on V2 samplers and estimators, consider using a different backend.")

        cost_h = formatter(problem)

        if cost_h is None:
            raise ValueError("Formatter returned None")

        self.n_qubits = cost_h.num_qubits

        best_sample, betas, energies, depths, cnot_counts = self._falqon_subroutine(cost_h, backend)

        best_data: BitArray = best_sample[0].data.meas
        counts: dict = best_data.get_counts()
        shots = best_data.num_shots

        result = {'betas': betas,
                  'energies': energies,
                  'depths': depths,
                  'cxs': cnot_counts,
                  'n': self.max_reps,
                  'delta_t': self.delta_t,
                  'beta_0': self.beta_0,
                  'energy': min(energies),
                  }

        return Result(
            best_bitstring=max(counts, key=counts.get),
            most_common_bitstring=max(counts, key=counts.get),
            distribution={k: v/shots for k, v in counts.items()},
            energies=energies,
            energy_std=np.std(energies),
            best_energy=min(energies),
            num_of_samples=shots,
            average_energy=np.mean(energies),
            most_common_bitstring_energy=0,
            result=result
        )

    def _add_ansatz_part(
        self,
        cost_hamiltonian: SparsePauliOp,
        driver_hamiltonian: SparsePauliOp,
        beta: float,
        circuit: QuantumCircuit
    ) -> None:
        """Adds a single FALQON ansatz 'building block' with the specified beta to the circuit"""
        circ_part = QuantumCircuit(circuit.num_qubits)

        circ_part.append(PauliEvolutionGate(cost_hamiltonian, time=self.delta_t), circ_part.qubits)
        circ_part.append(PauliEvolutionGate(driver_hamiltonian, time=self.delta_t * beta), circ_part.qubits)

        circuit.compose(circ_part, circ_part.qubits, inplace=True)

    def _build_ansatz(self, cost_hamiltonian, driver_hamiltonian, betas):
        """Build the FALQON circuit for the given betas"""

        circ = QuantumCircuit(self.n_qubits)
        circ.h(range(self.n_qubits))

        for beta in betas:
            circ.append(PauliEvolutionGate(cost_hamiltonian, time=self.delta_t), circ.qubits)
            circ.append(PauliEvolutionGate(driver_hamiltonian, time=self.delta_t * beta), circ.qubits)
        return circ

    def _falqon_subroutine(
            self,
            cost_hamiltonian: SparsePauliOp,
            backend: QiskitBackend
    ) -> tuple[PrimitiveResult[SamplerPubResult], list[float], list[float], list[int], list[int]]:
        """
        Run the 'meat' of the algorithm.

        Args:
            cost_hamiltonian (SparsePauliOp): Cost hamiltonian from the formatter.
            backend (QiskitBackend): Backend

        Returns:
            tuple[PrimitiveResult[SamplerPubResult], list[float], list[float], list[int], list[int]]: 
            Sampler result from best betas, list of betas, list of energies, list of depths, list of cnot counts
        """

        if self.driver_h is None:
            self.driver_h = SparsePauliOp.from_sparse_list([("X", [i], 1) for i in range(self.n_qubits)], num_qubits=self.n_qubits)
            driver_hamiltonian = self.driver_h
        else:
            driver_hamiltonian = self.driver_h

        hamiltonian_commutator = complex(0, 1) * commutator(driver_hamiltonian, cost_hamiltonian)

        betas = [self.beta_0]
        energies = []
        cnot_counts = []
        circuit_depths = []

        circuit = QuantumCircuit(self.n_qubits)
        circuit.h(circuit.qubits)

        self._add_ansatz_part(cost_hamiltonian, driver_hamiltonian, self.beta_0, circuit)

        for i in range(self.max_reps):

            beta = -1 * backend.estimator.run([(circuit, hamiltonian_commutator)]).result()[0].data.evs
            betas.append(beta)

            self._add_ansatz_part(cost_hamiltonian, driver_hamiltonian, beta, circuit)

            energy = backend.estimator.run([(circuit, cost_hamiltonian)]).result()[0].data.evs
            # print(i, energy)
            energies.append(energy)
            circuit_depths.append(circuit.depth())
            cnot_counts.append(circuit.count_ops().get('cx', 0))

        argmin = np.argmin(np.asarray(energies))

        sampling_circuit = self._build_ansatz(cost_hamiltonian, driver_hamiltonian, betas[:argmin])
        sampling_circuit.measure_all()

        best_sample = backend.sampler.run([(sampling_circuit)]).result()

        return best_sample, betas, energies, circuit_depths, cnot_counts
