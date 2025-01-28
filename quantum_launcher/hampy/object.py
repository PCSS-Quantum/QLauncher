from qiskit.quantum_info import SparsePauliOp
from typing import overload


class HampyEquation:
    @overload
    def __init__(self, size: int): ...
    @overload
    def __init__(self, hamiltonian: SparsePauliOp): ...
    @overload
    def __init__(self, sparse_list: list[tuple], size: int): ...

    def __init__(self, argument, *args):
        if isinstance(argument, int):
            self.size = argument
            self._hamiltonian = SparsePauliOp.from_sparse_list([('I', [], 0)], argument)
        elif isinstance(argument, SparsePauliOp):
            self.size = argument.num_qubits
            self._hamiltonian = argument
        elif isinstance(argument, list) and len(args) > 0 and isinstance(args[0], int):
            self.size = args[0]
            self._hamiltonian = SparsePauliOp.from_sparse_list(argument, args[0])
        else:
            raise TypeError('Wrong arguments!')

    def get_variable(self, index: int) -> "HampyVariable":
        assert isinstance(index, int), "Index needs to be an integer"
        obj = HampyVariable(index, self)
        return obj

    @property
    def hamiltonian(self) -> SparsePauliOp:
        return self._hamiltonian.simplify()

    @hamiltonian.setter
    def hamiltonian(self, new_hamiltonian: SparsePauliOp):
        self._hamiltonian = new_hamiltonian

    def to_sparse_pauli_op(self) -> SparsePauliOp:
        return self.hamiltonian

    def get_order(self) -> int:
        equation_order = 0
        for Z_term in self.hamiltonian.paulis:
            equation_order = max(equation_order, Z_term.count('Z'))
        return equation_order

    def is_quadratic(self) -> bool:
        return all(term.z.sum() <= 2 for term in self.hamiltonian.paulis)

    def __or__(self, other: "HampyVariable" | "HampyEquation", /) -> "HampyEquation":
        if isinstance(other, HampyVariable):
            other = other.to_equation()

        return HampyEquation(self.hamiltonian - other.hamiltonian - self.hamiltonian.compose(other.hamiltonian))

    def __and__(self, other: "HampyVariable" | "HampyEquation", /) -> "HampyEquation":
        if isinstance(other, HampyVariable):
            other = other.to_equation()

        return HampyEquation(self.hamiltonian.compose(other.hamiltonian))

    def __xor__(self, other: "HampyVariable" | "HampyEquation", /) -> "HampyEquation":
        if isinstance(other, HampyVariable):
            other = other.to_equation()

        return HampyEquation(self.hamiltonian - other.hamiltonian - (2 * self.hamiltonian.compose(other.hamiltonian)))

    def __invert__(self) -> "HampyEquation":
        I = ('I', [], 1)
        identity = SparsePauliOp.from_sparse_list([I], self.size)
        return HampyEquation(identity - self.hamiltonian)

    def __getitem__(self, variable_number: int):
        return self.get_variable(variable_number)

    def __eq__(self, other: "HampyEquation") -> bool:
        return self.hamiltonian == other.hamiltonian


class HampyVariable:
    def __init__(self, index: int, parent: HampyEquation):
        self.index = index
        self.size = parent.size

    def __xor__(self, other: "HampyVariable" | HampyEquation, /) -> HampyEquation:
        if isinstance(other, HampyEquation):
            return self.to_equation() ^ other

        I = ('I', [], 0.5)
        Z_term = ('ZZ', [self.index, other.index], -0.5)
        eq = HampyEquation(SparsePauliOp.from_sparse_list([I, Z_term], self.size))
        return eq

    def __or__(self, other: "HampyVariable" | HampyEquation, /) -> HampyEquation:
        if isinstance(other, HampyEquation):
            return self.to_equation() | other

        I_term = ('I', [], 0.25)
        Z1_term = ('Z', [self.index], -0.25)
        Z2_term = ('Z', [other.index], -0.25)
        ZZ_term = ('ZZ', [self.index, other.index], 0.25)
        eq = HampyEquation([I_term, Z1_term, Z2_term, ZZ_term], self.size)
        return eq

    def __and__(self, other: "HampyVariable" | HampyEquation, /) -> HampyEquation:

        if isinstance(other, HampyEquation):
            return self.to_equation() & other

        I_term = ('I', [], 0.75)
        Z1_term = ('Z', [self.index], -0.25)
        Z2_term = ('Z', [other.index], -0.25)
        ZZ_term = ('ZZ', [self.index, other.index], -0.25)
        eq = HampyEquation([I_term, Z1_term, Z2_term, ZZ_term], self.size)
        return eq

    def __invert__(self) -> HampyEquation:
        I_term = ('I', [], 0.5)
        Z_term = ('Z', [self.index], 0.5)
        return HampyEquation([I_term, Z_term], self.size)

    def to_equation(self) -> HampyEquation:
        I_term = ('I', [], 0.5)
        Z_term = ('Z', [self.index], -0.5)
        return HampyEquation([I_term, Z_term], self.size)
