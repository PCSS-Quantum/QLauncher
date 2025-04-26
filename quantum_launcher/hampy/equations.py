"""
`equations` module provides additional binary operations for Hampy objects.

It's goal is too simplify the creation of more complex problem implementations, by creating them with use of smaller ones.
"""
from typing import Optional
from copy import copy
from qiskit.quantum_info import SparsePauliOp
from .object import Equation, Variable


def one_in_n(variables: list[int | Variable], size: Optional[int] = None, quadratic: bool = False) -> Equation:
    """
    Generates Equation for One in N problem.

    One in N returns True if and only if exactly one of targeted indexes in 1, and all others are 0.

    Args:
        variables (list[int  |  Variable]): Triggered variables or variable indexes
        size (Optional[int], optional): Size of problem, if not given it takes the first found Variable.size value. Defaults to None.

    Returns:
        Equation: Equation returning True if exactly one of passed indexes is 1, False otherwise
    """
    if size is None:
        for var in variable:
            if isinstance(var, Variable):
                size = var.size
                break

    eq = Equation(size)
    new_variables = set()
    for var in copy(variables):
        if isinstance(var, int):
            new_variables.add(eq.get_variable(var))
        elif isinstance(var, Variable):
            new_variables.add(eq.get_variable(var.index))

    if quadratic:
        for variable in new_variables:
            eq += variable
        I = SparsePauliOp.from_sparse_list([('I', [], 1)], size)
        hamiltonian = I - eq.hamiltonian
        return Equation(hamiltonian.compose(hamiltonian))

    for variable in new_variables:
        equation = variable
        for different_var in new_variables - {variable}:
            equation &= ~different_var
        eq |= equation

    return eq
