from collections.abc import Callable 
from typing import TYPE_CHECKING, Any, Union, List

import numpy as np
from pyqubo import Spin
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.circuit.library import ZGate, UnitaryGate, QFTGate 
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
		return QUBO(qubo.quadratic.to_array(), 0)


class BQM(ProblemLike):
	def __init__(self, bqm: Any, model: Any = None) -> None:  # noqa: ANN401
		self.bqm = bqm


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

class GroverCircuit(ProblemLike):
    @staticmethod
    def create_oracle_from_bitstring(bit_string: str) -> Gate:
        """
        Creates oracle from given bit string
        """
        num_qubits = len(bit_string)
        qc = QuantumCircuit(num_qubits, name=f"Oracle_{bit_string}")

        reversed_s = bit_string[::-1] # Does this stay???

        for i, char in enumerate(reversed_s):
            if char == '0':
                qc.x(i)

        if num_qubits > 1:
            mcz = ZGate().control(num_qubits - 1)
            qc.append(mcz, range(num_qubits))
        else:
            qc.z(0)

        for i, char in enumerate(reversed_s):
            if char == '0':
                qc.x(i)

        return qc.to_gate()
   
    @staticmethod
    def _create_hadamard_walsh_transform(num_qubits):
        print("state_prep is None. Using Hadamard-Walsh Transform")
        state_prep_circ = QuantumCircuit(num_qubits)
        state_prep_circ.h(range(num_qubits))
        return state_prep_circ.to_gate()
   
    @staticmethod
    def _validate_and_create_oracle(val):
        if not all(c in '01' for c in val):
            raise ValueError("String/List must contain only zeros and ones.")
        return GroverCircuit.create_oracle_from_bitstring(val)
    
    def __init__(
        self, 
        oracle: QuantumCircuit | np.ndarray | List[str] | str,
        num_solutions: int = None,
        num_iterations: int = None,
        state_prep: QuantumCircuit | Gate | np.ndarray = None,
        num_state_qubits: int = None
        ):
        '''
        
        '''
        self.num_solutions = num_solutions
        
        if not(num_solutions or num_iterations):
            raise ValueError(f"At least one of num_solutions, num_iterations has to be not None")
    
        theta = np.arcsin(np.sqrt(num_solutions / (2**(oracle.num_qubits - 1))))
        self.num_iterations = num_iterations if num_iterations is not None else int(np.round((np.pi / (4 * theta))))
        
        self.state_prep = state_prep 
        
        self._gate = None        
        
        oracle_conversion_map= {
            QuantumCircuit: lambda x: x.to_gate(),
            str: GroverCircuit._validate_and_create_oracle,
            list: GroverCircuit._validate_and_create_oracle,
            np.ndarray: lambda x: UnitaryGate(x)
        }
        
        if type(oracle) in oracle_conversion_map:
            self.oracle = oracle_conversion_map[type(oracle)](oracle)
        else:
            raise TypeError(f"Unsupported data type: {type(oracle)}")

        self.num_state_qubits = num_state_qubits if not None else self.oracle.num_qubits - 1
        
        state_prep_strategies = {
            QuantumCircuit: lambda x: x.to_gate(),
            np.ndarray: lambda x: UnitaryGate(x),
            type(None): lambda _: GroverCircuit.create_hadamard_walsh_transform(self.num_state_qubits)
        }
        
        if type(state_prep) in state_prep_strategies:
            self.state_prep = state_prep_strategies[type(state_prep)](state_prep)
        elif not state_prep:
            self.state_prep = GroverCircuit.create_hadamard_walsh_transform(self.num_state_qubits)
            
        self.num_qubits = self.oracle.num_qubits
        
class ControlledModularMultiplierGates(ProblemLike):
    @staticmethod
    def create_controlled_modular_multiplier_gate(modulo, factor) -> Gate:
        
        n_modulo_number_qubits = np.ceil(np.log2(modulo))
        
        def create_adder_gate(factor, modulo):
            
            n_addition_qubits = n_modulo_number_qubits + 2
            
            phase_adder_circuit = QuantumCircuit(n_addition_qubits)
            
            for qubit_ix in range(n_modulo_number_qubits):
                phi = 2 * np.pi * factor / 2**(n_modulo_number_qubits - qubit_ix)
                phase_adder_circuit.rz(phi, qubit_ix)
                
            return phase_adder_circuit.to_gate()
        
        def create_modular_adder_gate(factor, modulo):
            
            factor_adder = create_adder_gate(factor, modulo)
            modulo_adder = create_adder_gate(modulo, modulo).inverse()
            
            controlled_modulo_adder = modulo_adder.control(1)
            
            factor_substractor = factor_adder.inverse()
            modulo_substractor = modulo_adder.inverse()
            
            n_qubits = n_modulo_number_qubits + 2
            
            qft = QFTGate(num_qubits=n_qubits - 1)
            qft_i = qft.inverse()
            
            y = QuantumRegister(n_modulo_number_qubits)
            qft_ancilla = QuantumRegister(1)
            ancilla = QuantumRegister(1)
            
            modular_adder_circuit = QuantumCircuit(n_qubits)
            
            modular_adder_circuit.append(factor_adder, y)
            modular_adder_circuit.append(modulo_substractor, y)
            modular_adder_circuit.append(qft_i, (y, qft_ancilla))
            modular_adder_circuit.cx(qft_ancilla, ancilla)
            modular_adder_circuit.append(qft, (y, qft_ancilla))
            modular_adder_circuit.append(controlled_modulo_adder, (ancilla, y))
            modular_adder_circuit.append(factor_substractor, y)
            modular_adder_circuit.append(qft_i, (y, qft_ancilla))
            modular_adder_circuit.x(qft_ancilla)
            modular_adder_circuit.cx(qft_ancilla, ancilla)
            modular_adder_circuit.x(qft_ancilla)
            modular_adder_circuit.append(qft, (y, qft_ancilla))
            modular_adder_circuit.append(factor_adder, y)
            
            return modular_adder_circuit.to_gate()
            
        def create_modular_multiplier_gate(factor, modulo):
            qft = QFTGate(num_qubits=n_modulo_number_qubits)
            qft_i = qft.inverse()
            
            
            x = QuantumRegister(n_modulo_number_qubits)
            y = QuantumRegister(n_modulo_number_qubits)
            ancillas = QuantumRegister(2)
            
            modular_multiplier_circuit = QuantumCircuit(x, y, ancillas)
            
            modular_multiplier_circuit.append(qft, y)
            
            for qubit_ix in range(n_modulo_number_qubits):
                controlled_modular_adder_gate = create_modular_adder_gate((2**qubit_ix * factor), modulo) # can cause wrong results - to be tested. Alternative is to add "% modulo"
                modular_multiplier_circuit.append(controlled_modular_adder_gate, (x[qubit_ix], y))
                
            modular_multiplier_circuit.append(qft_i, y)
            
            return modular_multiplier_circuit.to_gate()
        
        controlled_modular_multiplier_gate = create_modular_multiplier_gate(factor, modulo).control(1)
        controlled_modular_multiplier_gate_i = controlled_modular_multiplier_gate.inverse()
        
        control = QuantumRegister(1)
        x = QuantumRegister(n_modulo_number_qubits)
        y_with_ancillas = QuantumRegister(n_modulo_number_qubits + 2)
    
        controlled_modular_multiplier_gate_circuit = QuantumCircuit(control, x, y_with_ancillas)
        
        controlled_modular_multiplier_gate_circuit.append(controlled_modular_multiplier_gate, (control, x, y_with_ancillas))
        
        for qubit_ix in range(n_modulo_number_qubits):
            controlled_modular_multiplier_gate_circuit.swap(x[qubit_ix], y_with_ancillas[qubit_ix]) # ancillas are 0 so it's ok
        
        controlled_modular_multiplier_gate_circuit.append(controlled_modular_multiplier_gate_i, (control, x, y_with_ancillas))
        
        return controlled_modular_multiplier_gate_circuit.to_gate()
    
    def __init__(self, factor: int, modulo: int, circuits: list[QuantumCircuit | np.ndarray | Gate] = None, eigen_state_prep: QuantumCircuit | np.ndarray | Gate = None):
        self.factor = factor
        self.modulo = modulo
        self.gates = []
        
        if circuits is None:
            for exp in range(np.ceil(np.log2(modulo))):
                tmp_factor = (factor * 2 ** exp) % modulo
                self.gates.append(ModularMultiplierGate.create_controlled_modular_multiplier_gate(tmp_factor, modulo))
        else: 
            conversion_map = {
                QuantumCircuit: lambda c: c.to_gate(),
                np.ndarray: lambda c: UnitaryGate(c),
                Gate: lambda c: c
            }
            
            for circuit in circuits:
                target_type = type(circuit)
                
                if target_type in conversion_map:
                    self.gates.append(conversion_map[target_type](circuit))
                else:
                    raise TypeError(f"Unsupported data type: {target_type}")
        
        self.num_qubits = self.gates[0].num_qubits
        
        self.eigen_state_prep = eigen_state_prep
        
        target_type = type(eigen_state_prep)
        if self.eigen_state_prep and target_type in conversion_map:
            self.eigen_state_prep = conversion_map[target_type](self.eigen_state_prep)
        else:
            raise TypeError(f"Unsupported data type: {target_type}")
            
            
        