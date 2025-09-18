""" Algorithms for Qiskit routines """
from datetime import datetime
from collections.abc import Callable, Iterable
from typing import Any, Literal
import statistics
import numpy as np
from scipy.optimize import minimize

import qiskit_algorithms
from qiskit_algorithms import optimizers
from qiskit_algorithms.minimum_eigensolvers.diagonal_estimator import _evaluate_sparsepauli as evaluate_energy

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate, QAOAAnsatz, efficient_su2

from qiskit.primitives import PrimitiveResult, SamplerPubResult, BaseSamplerV1, BaseEstimatorV1
from qiskit.primitives.containers import BitArray
from qiskit.primitives.base.base_primitive import BasePrimitive

from qiskit.quantum_info import SparsePauliOp

from qiskit_nature.second_q.algorithms.ground_state_solvers import GroundStateEigensolver
from qiskit_nature.second_q.problems import EigenstateResult

from qlauncher.base import Problem, Algorithm, Result
from qlauncher.base.base import Backend
from qlauncher.routines.cirq import CirqBackend
from qlauncher.routines.qiskit.backends.qiskit_backend import QiskitBackend
from qlauncher.problems import Molecule


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


def int_to_bitstring(number: int, total_bits: int):
    return np.binary_repr(number, total_bits)[::-1]


def cvar_cost(probs_values: Iterable[tuple[float, float]], alpha: float):
    """CVar cost function to be used instead of mean for qaoa training"""
    if not (alpha > 0 and alpha <= 1):
        raise ValueError("Alpha must be in range (0,1]")
    cvar, acc = 0.0, 0.0
    for p, v in sorted(probs_values, key=lambda x: x[1]):
        if acc >= alpha:
            break
        acc += p
        cvar += v * min(p, alpha-acc)
    return cvar / alpha


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

    def __init__(
            self,
            p: int = 1,
            optimization_method: Literal['COBYLA'] = "COBYLA",
            max_evaluations: int = 100,
            training_aggregation_method: Literal['mean', 'cvar'] = 'mean',
            cvar_alpha: float = 1,
            alternating_ansatz: bool = False,
            aux=None,
            **alg_kwargs):
        super().__init__(**alg_kwargs)
        self.name: str = 'qaoa'
        self.aux = aux
        self.p: int = p

        self.optimization_method = optimization_method
        self.max_evaluations = max_evaluations
        self.training_aggregation_method = training_aggregation_method
        self.cvar_alpha = cvar_alpha

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

    def _get_optimized_circuit_params(self, circuit: QuantumCircuit, hamiltonian: SparsePauliOp, backend: QiskitBackend | CirqBackend) -> tuple[np.ndarray, list[float]]:
        """
        Optimize circuit params

        Args:
            circuit (QuantumCircuit): QAOA circuit to be optimized
            backend (Backend): Backend containing sampler

        Returns:
            tuple[QuantumCircuit,list[float]]: Circuit with optimal param values applied, energy history
        """
        costs = []

        def cost_fn(params: np.ndarray):
            job = backend.sampler.run([(circuit, params)])
            results = job.result()[0].data.meas.get_int_counts()
            shots = sum(results.values())

            probs_with_costs = {state: (count/shots, np.real(evaluate_energy(state, hamiltonian)))
                                for state, count in results.items()}

            cost = (sum(prob*cost for prob, cost in probs_with_costs.values())
                    if self.training_aggregation_method == 'mean' else
                    cvar_cost(probs_with_costs.values(), self.cvar_alpha))
            costs.append(cost)
            return cost

        res = minimize(
            cost_fn,
            np.array([np.pi]*(len(circuit.parameters)//2) + [np.pi / 2]*(len(circuit.parameters)//2)),
            method=self.optimization_method,
            tol=1e-2,
            options={
                'maxiter': self.max_evaluations
            }
        )

        return res.x, costs

    def run(self, problem: Problem, backend: Backend, formatter: Callable) -> Result:
        """ Runs the QAOA algorithm """

        if not (isinstance(backend, QiskitBackend) or isinstance(backend, CirqBackend)):
            raise ValueError('Backend should be CirqBackend, QiskitBackend or subclass.')

        hamiltonian: SparsePauliOp = formatter(problem)

        if self.alternating_ansatz:
            if self.mixer_h is None:
                self.mixer_h = formatter.get_mixer_hamiltonian(problem)
            if self.initial_state is None:
                self.initial_state = formatter.get_QAOAAnsatz_initial_state(
                    problem)

        # Cirq translation issues if we use QAOAAnsatz() by itself without appending it to a QuantumCircuit
        circuit = QuantumCircuit(hamiltonian.num_qubits)
        circuit.append(QAOAAnsatz(cost_operator=hamiltonian, reps=self.p).to_instruction(), range(hamiltonian.num_qubits))

        circuit.measure_all()

        opt_params, costs = self._get_optimized_circuit_params(circuit, hamiltonian, backend)

        job = backend.sampler.run([(circuit, opt_params)])
        results = job.result()[0].data.meas.get_int_counts()

        final_energies = {
            int_to_bitstring(k, circuit.num_qubits): np.real(evaluate_energy(k, hamiltonian))
            for k in results.keys()
        }
        final_counts = {int_to_bitstring(k, circuit.num_qubits): v for k, v in results.items()}

        depth = circuit.decompose(reps=10).depth()
        if 'cx' in circuit.decompose(reps=10).count_ops():
            cx_count = circuit.decompose(reps=10).count_ops()['cx']
        else:
            cx_count = 0

        return self.construct_result(
            {
                'energy': min(costs),
                'depth': depth,
                'cx_count': cx_count,
                'qpu_time': 0,
                'training_costs': costs,
                'final_sample_energies': final_energies,
                'final_sample_counts': final_counts,
                'optimal_point': opt_params  # needed for educated_guess
            })

    def construct_result(self, result: dict) -> Result:
        counts, energies = result['final_sample_counts'], result['final_sample_energies']
        num_of_samples = sum(counts.values())
        average_energy = statistics.mean(energies.values())
        energy_std = statistics.stdev(energies.values()) if len(energies) > 1 else 0

        best_bs = min(energies, key=energies.get)
        most_common_bs = max(counts, key=counts.get)

        return Result(
            best_bitstring=best_bs,
            best_energy=min(energies.values()),
            most_common_bitstring=most_common_bs,
            most_common_bitstring_energy=energies[most_common_bs],
            distribution={k: v/num_of_samples for k, v in counts.items()},
            energies=energies,
            num_of_samples=num_of_samples,
            average_energy=average_energy,
            energy_std=energy_std,
            result=result
        )


class FALQON(QiskitOptimizationAlgorithm):
    """
    Algorithm class with FALQON.

    Args:
        driver_h (Operator | None): The driver Hamiltonian for the problem.
        delta_t (float): The time step for the evolution operators.
        beta_0 (float): The initial value of beta.
        n (int): The number of iterations to run the algorithm.
        **alg_kwargs: Additional keyword arguments for the base class.

    Attributes:
        driver_h (Operator | None): The driver Hamiltonian for the problem.
        delta_t (float): The time step for the evolution operators.
        beta_0 (float): The initial value of beta.
        n (int): The number of iterations to run the algorithm.
        cost_h (Operator | None): The cost Hamiltonian for the problem.
        n_qubits (int): The number of qubits in the problem.
        parameters (list[str]): The list of algorithm parameters.

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


class VQE(QiskitOptimizationAlgorithm):
    """Variational Quantum EigenSolver - qiskit-algorithm implementation wrapper.

    Args:
        optimizer (optimizers.Optimizer | None, optional): Optimizer for VQE. Defaults to None.
        ansatz (QuantumCircuit | None, optional): VQE's ansatz. Defaults to None.
        with_numpy (bool, optional): Ignores ansatz parameter and backend, and changes solver to Numpy based. Defaults to False.
    """
    # pip install git+https://github.com/qiskit-community/qiskit-nature.git
    # pyscf

    def __init__(self, optimizer: optimizers.Optimizer | None = None,
                 ansatz: QuantumCircuit | None = None, with_numpy: bool = False) -> None:
        self.optimizer = optimizers.COBYLA() if optimizer is None else optimizer
        self.ansatz = ansatz
        self.num_qubits: int = 0
        self.with_numpy: bool = with_numpy
        super().__init__()

    @property
    def ansatz(self) -> QuantumCircuit:
        if self._ansatz is None:
            return efficient_su2(self.num_qubits)
        return self._ansatz

    @ansatz.setter
    def ansatz(self, custom_ansatz):
        self._ansatz = custom_ansatz

    def run(self, problem: Problem, backend: Backend, formatter: Callable[..., Any]) -> Result:
        if not isinstance(backend, QiskitBackend):
            raise ValueError('Backend should be QiskitBackend or subclass.')
        if not isinstance(problem, Molecule):
            raise ValueError('The problem for this algorithm should be Molecule problem')
        if not isinstance(problem.operator.num_qubits, int):
            raise ValueError('num_qubits from problem operator is expected to be int')

        estimator = backend.estimator
        self.num_qubits = problem.operator.num_qubits
        if self.with_numpy:
            solver = qiskit_algorithms.NumPyMinimumEigensolver()
        else:
            solver = qiskit_algorithms.VQE(estimator, self.ansatz, self.optimizer)
        vqe_gss = GroundStateEigensolver(problem.mapper, solver)
        vqe_results = vqe_gss.solve(problem.problem)
        return self.construct_result(vqe_results)

    def construct_result(self, result: EigenstateResult) -> Result:
        energy = result.total_energies[0]
        # Not the cleanest way
        return Result(
            '',
            energy,
            '',
            energy,
            {'': 1},
            {'': energy},
            1,
            energy,
            0,
            None
        )
