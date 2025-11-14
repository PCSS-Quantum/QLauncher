from qiskit.quantum_info import SparsePauliOp

from qlauncher.hampy.debug import TruthTable
from qlauncher.hampy.object import Equation


def test_with_sparse_pauli():
	sparse_list = [('I', [], 0.25), ('Z', [0], -0.25), ('Z', [1], -0.25), ('ZZ', [0, 1], 0.25)]

	hamiltonian = SparsePauliOp.from_sparse_list(sparse_list, 2)

	tt = TruthTable(hamiltonian)
	assert tt.check_if_binary()
	assert tt.size == 2
	assert len(tt.truth_table.values()) == 2**2
	assert tt.check_if_binary()


def test_with_hampy_object():
	equation = Equation(4)
	equation = equation[0] & ~equation[1] & (equation[2] | equation[3])
	tt = TruthTable(equation)
	assert tt.check_if_binary()
