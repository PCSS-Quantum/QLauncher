""" Hampy objects implementation """
from typing import overload
from qiskit.quantum_info import SparsePauliOp


class Equation:
    """Equation object is responsible for arranging group of variables and making operations between them.
    You can use equation to avoid grouping up the variables object as well as for optimization purposes.

    Raises:
        TypeError: _description_

    Returns:
        _type_: _description_
    """
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

    def get_variable(self, index: int) -> "Variable":
        """ return's variable object at the given index """
        assert isinstance(index, int), "Index needs to be an integer"
        obj = Variable(index, self)
        return obj

    @property
    def hamiltonian(self) -> SparsePauliOp:
        """ Returns simplified hamiltonian. """
        return self._hamiltonian.simplify()

    @hamiltonian.setter
    def hamiltonian(self, new_hamiltonian: SparsePauliOp):
        self._hamiltonian = new_hamiltonian

    def to_sparse_pauli_op(self) -> SparsePauliOp:
        """ Returns hamiltonian as SparsePauliOp. """
        return self.hamiltonian

    def get_order(self) -> int:
        """ Returns order of equation. """
        equation_order = 0
        for z_term in self.hamiltonian.paulis:
            equation_order = max(equation_order, str(z_term).count('Z'))
        return equation_order

    def is_quadratic(self) -> bool:
        """ Returns `True` if equation is up to quadratic """
        return all(term.z.sum() <= 2 for term in self.hamiltonian.paulis)

    def __or__(self, other: "Variable | Equation", /) -> "Equation":
        if isinstance(other, Variable):
            other = other.to_equation()

        return Equation(self.hamiltonian + other.hamiltonian - self.hamiltonian.compose(other.hamiltonian))

    def __and__(self, other: "Variable | Equation", /) -> "Equation":
        if isinstance(other, Variable):
            other = other.to_equation()

        return Equation(self.hamiltonian.compose(other.hamiltonian))

    def __xor__(self, other: "Variable | Equation", /) -> "Equation":
        if isinstance(other, Variable):
            other = other.to_equation()

        return Equation(self.hamiltonian + other.hamiltonian - (2 * self.hamiltonian.compose(other.hamiltonian)))

    def __invert__(self) -> "Equation":
        identity = ('I', [], 1)
        identity = SparsePauliOp.from_sparse_list([identity], self.size)
        return Equation(identity - self.hamiltonian)

    def __getitem__(self, variable_number: int):
        return self.get_variable(variable_number)

    def __eq__(self, other: "Equation") -> bool:
        if isinstance(other, Variable):
            other = other.to_equation()

        return self.hamiltonian == other.hamiltonian

    def __add__(self, other: "Variable | Equation") -> "Equation":
        if isinstance(other, Variable):
            other = other.to_equation()

        return Equation(self.hamiltonian + other.hamiltonian)

    def __radd__(self, other: "Equation") -> "Equation":
        if isinstance(other, Variable):
            other = other.to_equation()

        return Equation(self.hamiltonian + other.hamiltonian)

    def __mul__(self, other: "Equation | float") -> "Equation":
        if isinstance(other, Variable):
            other = other.to_equation()
        if isinstance(other, (int, float)):
            return Equation(float(other) * self.hamiltonian)
        return Equation(self.hamiltonian.compose(other.hamiltonian))

    def __rmul__(self, other: "Equation | float") -> "Equation":
        if isinstance(other, Variable):
            other = other.to_equation()
        if isinstance(other, (float, int)):
            return Equation(float(other) * self.hamiltonian)
        return Equation(self.hamiltonian.compose(other.hamiltonian))


class Variable:
    """Class for setting variables
    """

    def __init__(self, index: int, parent: Equation):
        self.index = index
        self.size = parent.size

    def __xor__(self, other: "Equation | float", /) -> Equation:
        if isinstance(other, Equation):
            return self.to_equation() ^ other

        identity = ('I', [], 0.5)
        z_term = ('ZZ', [self.index, other.index], -0.5)
        eq = Equation(SparsePauliOp.from_sparse_list([identity, z_term], self.size))
        return eq

    def __or__(self, other: "Variable | Equation", /) -> Equation:
        if isinstance(other, Equation):
            return self.to_equation() | other

        i_term = ('I', [], 0.75)
        z1_term = ('Z', [self.index], -0.25)
        z2_term = ('Z', [other.index], -0.25)
        zz_term = ('ZZ', [self.index, other.index], -0.25)
        eq = Equation([i_term, z1_term, z2_term, zz_term], self.size)
        return eq

    def __and__(self, other: "Variable | Equation", /) -> Equation:

        if isinstance(other, Equation):
            return self.to_equation() & other

        i_term = ('I', [], 0.25)
        z1_term = ('Z', [self.index], -0.25)
        z2_term = ('Z', [other.index], -0.25)
        zz_term = ('ZZ', [self.index, other.index], 0.25)
        eq = Equation([i_term, z1_term, z2_term, zz_term], self.size)
        return eq

    def __invert__(self) -> Equation:
        i_term = ('I', [], 0.5)
        z_term = ('Z', [self.index], 0.5)
        return Equation([i_term, z_term], self.size)

    def to_equation(self) -> Equation:
        """ Changes variable into equation """
        i_term = ('I', [], 0.5)
        z_term = ('Z', [self.index], -0.5)
        return Equation([i_term, z_term], self.size)
