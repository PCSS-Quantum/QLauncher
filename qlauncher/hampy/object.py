from typing import overload

from qiskit.quantum_info import SparsePauliOp


class Equation:
    """## Hampy Equation
    Represents a binary or general-purpose Hamiltonian equation built on
    :class:`qiskit.quantum_info.SparsePauliOp`.
    The class provides logical-style operators (AND, OR, XOR, NOT) interpreted
    as operations on Hamiltonians with binary energies {0, 1}, along with basic
    arithmetic operations and utilities for analyzing the resulting operator.

    Examples
    --------
    Create an empty equation of size 3:
    >>> eq = Equation(3)
    >>> eq.hamiltonian
    SparsePauliOp(['III'], coeffs=[0.])

    Create from a SparsePauliOp:
    >>> from qiskit.quantum_info import SparsePauliOp
    >>> H = SparsePauliOp.from_sparse_list([('Z', [0], 1.0)], 1)
    >>> eq = Equation(H)
    >>> eq.get_order()
    1

    Use variables and logical operators:
    >>> eq = Equation(2)
    >>> x0 = eq[0]
    >>> x1 = eq[1]

    XOR:
    >>> h_xor = x0 ^ x1
    >>> h_xor.hamiltonian
    SparsePauliOp([...])

    OR:
    >>> h_or = x0 | x1

    AND:
    >>> h_and = x0 & x1

    Negation:
    >>> h_not = ~x0

    Combine equations arithmetically:
    >>> h_sum = (x0 ^ x1) + (x0 & x1)
    >>> h_scaled = 2.0 * h_sum
    >>> h_divided = h_sum / 3

    Export back to SparsePauliOp:
    >>> H = h_sum.hamiltonian
    """

    @overload
    def __init__(self, size: int, /): ...
    @overload
    def __init__(self, hamiltonian: SparsePauliOp, /): ...
    @overload
    def __init__(self, sparse_list: list[tuple], size: int, /): ...

    def __init__(self, argument, *args):
        self.size: int
        if isinstance(argument, int):
            self.size = argument
            self._hamiltonian = SparsePauliOp.from_sparse_list([('I', [], 0)], argument)
        elif isinstance(argument, SparsePauliOp):
            if not isinstance(argument.num_qubits, int):
                raise TypeError('Cannot read number of qubits from provided SparsePauliOp')
            self.size = argument.num_qubits
            self._hamiltonian = argument
        elif isinstance(argument, list) and len(args) > 0 and isinstance(args[0], int):
            self.size = args[0]
            self._hamiltonian = SparsePauliOp.from_sparse_list(argument, args[0])
        else:
            raise TypeError('Wrong arguments!')

    def get_variable(self, index: int) -> 'Variable':
        return Variable(index, self.size)

    @property
    def hamiltonian(self) -> SparsePauliOp:
        return self._hamiltonian.simplify()

    @hamiltonian.setter
    def hamiltonian(self, new_hamiltonian: SparsePauliOp) -> None:
        self._hamiltonian = new_hamiltonian

    def to_sparse_pauli_op(self) -> SparsePauliOp:
        return self.hamiltonian

    def get_order(self) -> int:
        equation_order = 0
        for Z_term in self.hamiltonian.paulis:
            equation_order = max(equation_order, str(Z_term).count('Z'))
        return equation_order

    def is_quadratic(self) -> bool:
        return all(term.z.sum() <= 2 for term in self.hamiltonian.paulis)

    def __or__(self, other: 'Variable | Equation', /) -> 'Equation':
        if isinstance(other, Variable):
            other = other.to_equation()

        return Equation(self.hamiltonian + other.hamiltonian - self.hamiltonian.compose(other.hamiltonian))

    def __and__(self, other: 'Variable | Equation', /) -> 'Equation':
        if isinstance(other, Variable):
            other = other.to_equation()

        return self * other

    def __xor__(self, other: 'Variable | Equation', /) -> 'Equation':
        if isinstance(other, Variable):
            other = other.to_equation()

        return Equation(self.hamiltonian + other.hamiltonian - (2 * self.hamiltonian.compose(other.hamiltonian)))

    def __invert__(self) -> 'Equation':
        I_term = ('I', [], 1)
        identity = SparsePauliOp.from_sparse_list([I_term], self.size)
        return Equation(identity - self.hamiltonian)

    def __getitem__(self, variable_number: int):
        return self.get_variable(variable_number)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (Variable, Equation)):
            raise TypeError(f'Cannot compare hampy.Equation to {type(other)}')
        if isinstance(other, Variable):
            other = other.to_equation()

        return self.hamiltonian == other.hamiltonian

    def __add__(self, other: 'Variable | Equation | SparsePauliOp') -> 'Equation':
        if isinstance(other, SparsePauliOp):
            other = Equation(other)
        elif isinstance(other, Variable):
            other = other.to_equation()

        return Equation(self.hamiltonian + other.hamiltonian)

    def __radd__(self, other: 'Equation') -> 'Equation':
        if isinstance(other, Variable):
            other = other.to_equation()

        return Equation(self.hamiltonian + other.hamiltonian)

    def __mul__(self, other: 'Equation | float') -> 'Equation':
        if isinstance(other, Variable):
            other = other.to_equation()
        if isinstance(other, (float, int)):
            return Equation(float(other) * self.hamiltonian)
        return Equation(self.hamiltonian.compose(other.hamiltonian))

    def __rmul__(self, other: 'Equation | float') -> 'Equation':
        if isinstance(other, Variable):
            other = other.to_equation()
        if isinstance(other, (float, int)):
            return Equation(float(other) * self.hamiltonian)
        return Equation(self.hamiltonian.compose(other.hamiltonian))

    def __truediv__(self, other: float) -> 'Equation':
        return Equation(self.hamiltonian / other)


class Variable:
    def __init__(self, index: int, size: int):
        self.index = index
        self.size = size

    def __xor__(self, other: 'Equation | Variable', /) -> Equation:
        if isinstance(other, Equation):
            return self.to_equation() ^ other

        I_term = ('I', [], 0.5)
        Z_term = ('ZZ', [self.index, other.index], -0.5)
        return Equation(SparsePauliOp.from_sparse_list([I_term, Z_term], self.size))

    def __or__(self, other: 'Variable | Equation', /) -> Equation:
        if isinstance(other, Equation):
            return self.to_equation() | other

        I_term = ('I', [], 0.75)
        Z1_term = ('Z', [self.index], -0.25)
        Z2_term = ('Z', [other.index], -0.25)
        ZZ_term = ('ZZ', [self.index, other.index], -0.25)
        return Equation([I_term, Z1_term, Z2_term, ZZ_term], self.size)

    def __and__(self, other: 'Variable | Equation', /) -> Equation:
        if isinstance(other, Equation):
            return self.to_equation() & other

        I_term = ('I', [], 0.25)
        Z1_term = ('Z', [self.index], -0.25)
        Z2_term = ('Z', [other.index], -0.25)
        ZZ_term = ('ZZ', [self.index, other.index], 0.25)
        return Equation([I_term, Z1_term, Z2_term, ZZ_term], self.size)

    def __invert__(self) -> Equation:
        I_term = ('I', [], 0.5)
        Z_term = ('Z', [self.index], 0.5)
        return Equation([I_term, Z_term], self.size)

    def to_equation(self) -> Equation:
        I_term = ('I', [], 0.5)
        Z_term = ('Z', [self.index], -0.5)
        return Equation([I_term, Z_term], self.size)
