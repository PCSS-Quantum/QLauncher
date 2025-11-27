from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from pyqubo import Spin
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.mappers import ParityMapper, QubitMapper
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import from_ising

if TYPE_CHECKING:
	from qiskit_nature.second_q.problems import ElectronicStructureProblem


class ProblemLike:
	_all_problems: dict[str, type['ProblemLike']] = {}

	def __init__(self, instance: Any) -> None:
		self.instance = instance

	def __init_subclass__(cls) -> None:
		if ProblemLike not in cls.__bases__:
			return
		ProblemLike._all_problems[cls.__name__] = cls
		cls._mapping: dict[type[ProblemLike], Callable[[], ProblemLike]] = {}
		for method_name in cls.__dict__:
			if method_name.startswith('to_'):
				method = cls.__dict__[method_name]
				cls._mapping[method.__annotations__['return']] = method


class QUBO(ProblemLike):
	def __init__(self, matrix: np.ndarray, offset: float = 0) -> None:
		self.matrix = matrix
		self.offset = offset

	def to_hamiltonian(self) -> 'Hamiltonian':
		num_vars = self.matrix.shape[0]
		pauli = 0
		for i, col in enumerate(self.matrix):
			for j, entry in enumerate(col):
				if entry == 0:
					continue
				if i == j:
					pauli += SparsePauliOp.from_sparse_list([('I', [0], 0.5), ('Z', [i], -0.5)], num_vars) * entry
				else:
					pauli += (
						SparsePauliOp.from_sparse_list(
							[('I', [0], 0.25), ('Z', [i], -0.25), ('Z', [j], -0.25), ('ZZ', [i, j], 0.25)], num_vars
						)
						* entry
					)
		pauli += SparsePauliOp.from_sparse_list([('I', [], self.offset)], num_vars)
		return Hamiltonian(pauli)

	def to_fn(self) -> 'FN':
		def function(bin_vec: np.ndarray) -> float:
			return np.dot(bin_vec, np.dot(self.matrix, bin_vec)) + self.offset

		return FN(function)

	def to_bqm(self: 'QUBO') -> 'BQM':
		matrix = self.matrix
		symmetric = (self.matrix.transpose() == self.matrix).all()
		if not symmetric:
			for i in range(len(self.matrix)):
				for j in range(len(self.matrix)):
					if i > j:
						matrix[i][j] = 0

		values_and_qubits = {(x, y): c for y, r in enumerate(matrix) for x, c in enumerate(r) if c != 0}
		number_of_qubits = len(matrix)
		qubits = [Spin(f'x{i}') for i in range(number_of_qubits)]
		H = 0
		for (x, y), value in values_and_qubits.items():
			if symmetric:
				H += value / len({x, y}) * qubits[x] * qubits[y]
			else:
				H += value * qubits[x] * qubits[y]
		model = H.compile()
		bqm = model.to_bqm()
		bqm.offset += self.offset
		return BQM(bqm, model)


class FN(ProblemLike):
	def __init__(self, function: Callable[[np.ndarray], float]) -> None:
		self.function = function

	def __call__(self, vector: np.ndarray) -> float:
		return self.function(vector)


class Hamiltonian(ProblemLike):
	def __init__(
		self,
		hamiltonian: SparsePauliOp,
		mixer_hamiltonian: SparsePauliOp | None = None,
		initial_state: QuantumCircuit | None = None,
	) -> None:
		self.hamiltonian = hamiltonian
		self._mixer_hamiltonian: SparsePauliOp | None = mixer_hamiltonian
		self._initial_state: QuantumCircuit | None = initial_state

	@property
	def mixer_hamiltonian(self) -> SparsePauliOp | None:
		return self._mixer_hamiltonian

	@mixer_hamiltonian.setter
	def mixer_hamiltonian(self, mixer_hamiltonian: SparsePauliOp) -> None:
		self._mixer_hamiltonian = mixer_hamiltonian

	@property
	def initial_state(self) -> QuantumCircuit | None:
		return self._initial_state

	@initial_state.setter
	def initial_state(self, initial_state: QuantumCircuit) -> None:
		self._initial_state = initial_state

	def to_qubo(self) -> QUBO:
		qp = from_ising(self.hamiltonian)
		conv = QuadraticProgramToQubo()
		qubo = conv.convert(qp).objective
		return QUBO(qubo.quadratic.to_array(), qubo.constant)


class BQM(ProblemLike):
	def __init__(self, bqm: Any, model: Any = None) -> None:  # noqa: ANN401
		self.bqm = bqm
		self.model = model


class Molecule(ProblemLike):
	def __init__(self, instance: MoleculeInfo, mapper: QubitMapper | None = None, basis_set: str = 'STO-6G') -> None:
		self.instance = instance
		self.basis_set = basis_set
		self.mapper = ParityMapper() if mapper is None else mapper
		self.problem: ElectronicStructureProblem = PySCFDriver.from_molecule(instance, basis=self.basis_set).run()
		self.mapper.num_particles = self.problem.num_particles
		operator = self.mapper.map(self.problem.hamiltonian.second_q_op())
		if not isinstance(operator, SparsePauliOp):
			raise TypeError
		self.operator: SparsePauliOp = operator

	@staticmethod
	def from_preset(instance_name: str) -> 'Molecule':
		match instance_name:
			case 'H2':
				instance = MoleculeInfo(['H', 'H'], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.74)])
			case _:
				raise ValueError(f"Molecule {instance_name} not supported, currently you can use: 'H2'")
		return Molecule(instance)
