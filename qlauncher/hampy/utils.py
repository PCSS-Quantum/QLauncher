"""
Utility functions for Hampy objects.
"""

from qiskit.quantum_info import Pauli, SparsePauliOp

from qlauncher.hampy.object import Equation


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

	n_paulis = []
	n_coeffs = []

	for p_string, coeff in op.label_iter():
		p_string = p_string[shift:] + p_string[:shift]
		n_paulis.append(Pauli(data=p_string))
		n_coeffs.append(coeff)

	new_op = SparsePauliOp(n_paulis, coeffs=n_coeffs)

	return Equation(new_op)
