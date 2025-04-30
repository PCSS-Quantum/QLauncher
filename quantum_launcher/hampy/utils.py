"""
Utility functions for Hampy objects.
"""
from qiskit.quantum_info import SparsePauliOp, Pauli
from quantum_launcher.hampy.object import Equation


def shift_affected_qubits(equation: Equation, shift: int) -> Equation:
    """
    Shifts the qubits affected by the equation by the given amount, wrapping around if the index goes out of bounds.

    For each Pauli in the equation hamiltonian, shifts the Pauli string by the given amount.
    i.e (shift = 1) IZIZ -> ZIZI, etc. !Might be unwanted! ZIII -> IIIZ
    Keeps the coefficients the same.

    Args:
        equation: Equation to shift
        shift: Amount to shift by

    Returns:
        Equation: New equation affecting the shifted qubits
    """
    op = equation.hamiltonian

    if shift == 0:
        return equation

    npaulis = []
    ncoeffs = []

    for p_string, coeff in op.label_iter():
        p_string = p_string[shift:] + p_string[:shift]
        npaulis.append(Pauli(data=p_string))
        ncoeffs.append(coeff)

    new_op = SparsePauliOp(npaulis, coeffs=ncoeffs)

    return Equation(new_op)
