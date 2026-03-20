from collections.abc import Callable
from typing import TYPE_CHECKING, Any, overload

import numpy as np
from dimod import BinaryQuadraticModel
from pyqubo import Binary  # type: ignore
from pyqubo import Model as pyquboModel
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.circuit.library import QFTGate, SwapGate, UnitaryGate, XGate
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.mappers import ParityMapper, QubitMapper
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import from_ising

if TYPE_CHECKING:
    from qiskit_nature.second_q.problems import ElectronicStructureProblem

from qlauncher.hampy import Equation


class Model:
    _all_problems: dict[str, type['Model']] = {}

    def __init__(self, instance: Any) -> None:
        self.instance = instance

    def __init_subclass__(cls) -> None:
        if Model not in cls.__bases__:
            return
        Model._all_problems[cls.__name__] = cls
        cls._mapping: dict[type[Model], Callable[[], Model]] = {}
        for method_name in cls.__dict__:
            if method_name.startswith('to_'):
                method = cls.__dict__[method_name]
                cls._mapping[method.__annotations__['return']] = method


class QUBO(Model):
    def __init__(self, matrix: np.ndarray, offset: float = 0) -> None:
        self.matrix = (matrix + matrix.T) / 2
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
        pauli += SparsePauliOp.from_sparse_list([('I', [0], self.offset)], num_vars)
        return Hamiltonian(Equation(pauli))

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
                    elif j > j:
                        matrix[i][j] *= 2

        values_and_qubits = {(x, y): c for y, r in enumerate(matrix) for x, c in enumerate(r) if c != 0}
        number_of_qubits = len(matrix)
        qubits = [Binary(f'x{i}') for i in range(number_of_qubits)]
        H = 0
        for (x, y), value in values_and_qubits.items():
            H += value * qubits[x] * qubits[y]
        model = H.compile()
        return BQM(model)


class FN(Model):
    def __init__(self, function: Callable[[np.ndarray], float]) -> None:
        self.function = function

    def __call__(self, vector: np.ndarray) -> float:
        return self.function(vector)


class Hamiltonian(Model):
    @overload
    def __init__(
        self, hamiltonian: SparsePauliOp, mixer_hamiltonian: SparsePauliOp | None = None, initial_state: QuantumCircuit | None = None
    ) -> None: ...
    @overload
    def __init__(
        self,
        hamiltonian: Equation,
        mixer_hamiltonian: Equation | None = None,
        initial_state: QuantumCircuit | None = None,
    ) -> None: ...
    def __init__(
        self,
        hamiltonian: Equation | SparsePauliOp,
        mixer_hamiltonian: Equation | SparsePauliOp | None = None,
        initial_state: QuantumCircuit | None = None,
    ) -> None:
        if isinstance(hamiltonian, SparsePauliOp):
            hamiltonian = Equation(hamiltonian)
        if isinstance(mixer_hamiltonian, SparsePauliOp):
            mixer_hamiltonian = Equation(mixer_hamiltonian)
        self._hampy_equation = hamiltonian
        self._hampy_mixer_equation: Equation | None = mixer_hamiltonian
        self._initial_state: QuantumCircuit | None = initial_state

    @property
    def mixer_hamiltonian(self) -> SparsePauliOp | None:
        if self._hampy_mixer_equation is not None:
            return self._hampy_mixer_equation.hamiltonian
        return None

    @property
    def hamiltonian(self) -> SparsePauliOp:
        return self._hampy_equation.hamiltonian

    @property
    def is_quadratic(self) -> bool:
        return self._hampy_equation.is_quadratic()

    @mixer_hamiltonian.setter
    def mixer_hamiltonian(self, mixer_hamiltonian: SparsePauliOp) -> None:
        self._hampy_mixer_equation = Equation(mixer_hamiltonian)

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
        return QUBO(qubo.quadratic.to_array(), float(qubo.constant))


class BQM(Model):
    def __init__(self, model: pyquboModel) -> None:  # noqa: ANN401
        self.model = model

    @property
    def bqm(self) -> BinaryQuadraticModel:
        return self.model.to_bqm()

    def to_qubo(self) -> QUBO:
        """Returns Qubo function"""
        model = self.model
        variables = sorted(model.variables)
        num_qubits = len(variables)
        Q_matrix = np.zeros((num_qubits, num_qubits))
        inv_map = {v: i for i, v in enumerate(variables)}
        qubo_dict, offset = model.to_qubo()
        for key, value in qubo_dict.items():
            var1, var2 = key
            Q_matrix[inv_map[var1], inv_map[var2]] = value
        return QUBO(Q_matrix, offset)

    def to_hamiltonian(self) -> Hamiltonian:
        """Returns Hamiltonian function"""

        bqm = self.model.to_bqm()
        variables, new_offset = bqm.variables, bqm.offset
        variables = list(variables)
        variables.sort()

        sparse_list = []

        for i, coeff in bqm.linear.items():
            sparse_list.append(('Z', [variables.index(i)], -coeff / 2))
            new_offset += coeff / 2

        for (i, j), coeff in bqm.quadratic.items():
            sparse_list.append(('ZZ', [variables.index(i), variables.index(j)], coeff / 4))
            sparse_list.append(('Z', [variables.index(i)], -coeff / 4))
            sparse_list.append(('Z', [variables.index(j)], -coeff / 4))
            new_offset += coeff / 4

        sparse_list.append(('I', [0], new_offset))

        return Hamiltonian(SparsePauliOp.from_sparse_list(sparse_list, num_qubits=len(variables)).simplify())


class Molecule(Model):
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


class GroverCircuit(Model):
    @staticmethod
    def create_oracle_from_bitstring(bit_string: str | list[str]) -> Gate:
        """
        Creates oracle from given bit string
        """
        num_qubits = len(bit_string) + 1
        qc = QuantumCircuit(num_qubits, name=f'Oracle_{bit_string}')

        reversed_s = bit_string[::-1]  # Does this stay???

        qc.x(-1)
        qc.h(-1)

        for i, char in enumerate(reversed_s):
            if char == '0':
                qc.x(i)

        mcx = XGate().control(num_qubits - 1)
        qc.append(mcx, range(num_qubits))

        for i, char in enumerate(reversed_s):
            if char == '0':
                qc.x(i)

        qc.h(-1)
        qc.x(-1)

        return qc.to_gate()

    @staticmethod
    def _create_hadamard_walsh_transform(num_qubits: int) -> Gate:
        print('state_prep is None. Using Hadamard-Walsh Transform')
        state_prep_circ = QuantumCircuit(num_qubits)
        state_prep_circ.h(range(num_qubits))
        return state_prep_circ.to_gate()

    @staticmethod
    def _validate_and_create_oracle(val: str | list[str]) -> Gate:
        if not all(c in '01' for c in val):
            raise ValueError('String/List must contain only zeros and ones.')
        return GroverCircuit.create_oracle_from_bitstring(val)

    def __init__(
        self,
        oracle: QuantumCircuit | np.ndarray | list[str] | str,
        num_solutions: int = None,
        num_iterations: int = None,
        state_prep: QuantumCircuit | Gate | np.ndarray = None,
        num_state_qubits: int = None,
    ):
        """ """
        self.num_solutions = num_solutions

        if not (num_solutions or num_iterations):
            raise ValueError('At least one of num_solutions, num_iterations has to be not None')

        self.state_prep = state_prep

        self._gate = None

        oracle_conversion_map = {
            QuantumCircuit: lambda x: x.to_gate(),
            str: GroverCircuit._validate_and_create_oracle,
            list: GroverCircuit._validate_and_create_oracle,
            np.ndarray: lambda x: UnitaryGate(x),
        }

        if type(oracle) in oracle_conversion_map:
            self.oracle = oracle_conversion_map[type(oracle)](oracle)
        else:
            raise TypeError(f'Unsupported data type: {type(oracle)}')

        theta = np.arcsin(np.sqrt(num_solutions / (2 ** (self.oracle.num_qubits - 1))))
        self.num_iterations = num_iterations if num_iterations is not None else int(np.round(np.pi / (4 * theta)))

        self.num_state_qubits = num_state_qubits if num_state_qubits is not None else self.oracle.num_qubits - 1

        state_prep_conversion_map = {
            QuantumCircuit: lambda x: x.to_gate(),
            np.ndarray: lambda x: UnitaryGate(x),
            type(None): lambda _: GroverCircuit._create_hadamard_walsh_transform(self.num_state_qubits),
        }

        if type(self.state_prep) in state_prep_conversion_map:
            self.state_prep = state_prep_conversion_map[type(self.state_prep)](self.state_prep)

        self.num_qubits = self.oracle.num_qubits


class ShorCircuit(Model):
    @staticmethod
    def create_modular_multiplier_gate_with_uncomputation(factor: int, modulo: int) -> Gate:
        n_modulo_number_qubits = int(np.floor(np.log2(modulo))) + 1

        def create_adder_gate(factor: int, modulo: int) -> Gate:
            phase_adder_circuit = QuantumCircuit(n_modulo_number_qubits + 1)  # + 1 to avoid overflow

            for qubit_ix in range(n_modulo_number_qubits + 1):
                phi = 2 * np.pi * factor / 2 ** (n_modulo_number_qubits + 1 - qubit_ix)
                phase_adder_circuit.rz(phi, qubit_ix)

            return phase_adder_circuit.to_gate()

        def create_modular_adder_gate(factor: int, modulo: int) -> Gate:
            factor_adder = create_adder_gate(factor, modulo)
            modulo_adder = create_adder_gate(modulo, modulo)

            controlled_modulo_adder = modulo_adder.control(1)

            factor_substractor = factor_adder.inverse()
            modulo_substractor = modulo_adder.inverse()

            qft = QFTGate(num_qubits=n_modulo_number_qubits + 1)
            qft_i = qft.inverse()

            y = QuantumRegister(n_modulo_number_qubits + 1)
            ancilla = QuantumRegister(1)
            controls = QuantumRegister(2)

            modular_adder_circuit = QuantumCircuit(controls, y, ancilla)

            modular_adder_circuit.append(factor_adder.control(2), controls[:] + y[:])
            modular_adder_circuit.append(modulo_substractor, y)
            modular_adder_circuit.append(qft_i, y)
            modular_adder_circuit.cx(y[-1], ancilla)
            modular_adder_circuit.append(qft, y)
            modular_adder_circuit.append(controlled_modulo_adder, ancilla[:] + y[:])
            modular_adder_circuit.append(factor_substractor.control(2), controls[:] + y[:])
            modular_adder_circuit.append(qft_i, y)
            modular_adder_circuit.x(y[-1])
            modular_adder_circuit.cx(y[-1], ancilla)
            modular_adder_circuit.x(y[-1])
            modular_adder_circuit.append(qft, y[:])
            modular_adder_circuit.append(factor_adder.control(2), controls[:] + y[:])

            return modular_adder_circuit.to_gate()

        def create_modular_multiplier_gate(factor: int, modulo: int) -> Gate:
            qft = QFTGate(num_qubits=n_modulo_number_qubits + 1)
            qft_i = qft.inverse()

            x = QuantumRegister(n_modulo_number_qubits)
            y = QuantumRegister(n_modulo_number_qubits + 1)
            ancilla = QuantumRegister(1)
            control = QuantumRegister(1)

            modular_multiplier_circuit = QuantumCircuit(control, x, y, ancilla)

            modular_multiplier_circuit.append(qft, y)

            for qubit_ix in range(n_modulo_number_qubits):
                controlled_modular_adder_gate = create_modular_adder_gate(((2**qubit_ix * factor) % modulo), modulo)
                modular_multiplier_circuit.append(controlled_modular_adder_gate, control[:] + [x[qubit_ix]] + y[:] + ancilla[:])

            modular_multiplier_circuit.append(qft_i, y)

            return modular_multiplier_circuit.to_gate()

        controlled_modular_multiplier_gate = create_modular_multiplier_gate(factor, modulo)
        controlled_modular_multiplier_gate_i = create_modular_multiplier_gate(pow(factor, -1, modulo), modulo).inverse()

        control = QuantumRegister(1)
        x = QuantumRegister(n_modulo_number_qubits)
        y_with_ancilla = QuantumRegister(n_modulo_number_qubits + 1 + 1)

        controlled_modular_multiplier_gate_circuit = QuantumCircuit(control, x, y_with_ancilla)

        controlled_modular_multiplier_gate_circuit.append(controlled_modular_multiplier_gate, control[:] + x[:] + y_with_ancilla[:])

        controlled_swap_gate = SwapGate().control()

        for qubit_ix in range(n_modulo_number_qubits):
            controlled_modular_multiplier_gate_circuit.append(
                controlled_swap_gate, control[:] + [x[qubit_ix]] + [y_with_ancilla[qubit_ix]]
            )  # ancillas are 0 so it's ok

        controlled_modular_multiplier_gate_circuit.append(controlled_modular_multiplier_gate_i, control[:] + x[:] + y_with_ancilla[:])

        return controlled_modular_multiplier_gate_circuit.to_gate()

    def __init__(
        self,
        factor: int,
        modulo: int,
        n_phase_kickback_qubits: int = None,
        circuits: list[QuantumCircuit | np.ndarray | Gate] = None,
        eigen_state_prep: QuantumCircuit | np.ndarray | Gate = None,
    ):
        self.factor = factor
        self.modulo = modulo
        self.gates = []

        if not n_phase_kickback_qubits:
            self.n_phase_kickback_qubits = 2 * int(np.floor(np.log2(self.modulo) + 1))
        else:
            self.n_phase_kickback_qubits = n_phase_kickback_qubits

        conversion_map = {
            QuantumCircuit: lambda c: c.to_gate(),
            np.ndarray: lambda c: UnitaryGate(c),
            Gate: lambda c: c,
            type(None): lambda c: None,
        }

        if circuits is None:
            for exp in range(2 * (int(np.floor(np.log2(modulo))) + 1)):
                tmp_factor = (factor ** (2**exp)) % modulo
                self.gates.append(ShorCircuit.create_modular_multiplier_gate_with_uncomputation(tmp_factor, modulo))
        else:
            for circuit in circuits:
                target_type = type(circuit)

                if target_type in conversion_map:
                    self.gates.append(conversion_map[target_type](circuit))
                else:
                    raise TypeError(f'Unsupported data type: {target_type}')

        self.num_qubits = self.gates[0].num_qubits

        self.eigen_state_prep = eigen_state_prep

        target_type = type(eigen_state_prep)
        if target_type in conversion_map:
            self.eigen_state_prep = conversion_map[target_type](self.eigen_state_prep)
        else:
            raise TypeError(f'Unsupported data type: {target_type}')
