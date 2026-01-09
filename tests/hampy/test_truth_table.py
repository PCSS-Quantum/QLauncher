from qiskit.quantum_info import SparsePauliOp

import qlauncher.hampy as hp


def test_with_sparse_pauli() -> None:
	sparse_list = [('I', [], 0.25), ('Z', [0], -0.25), ('Z', [1], -0.25), ('ZZ', [0, 1], 0.25)]

	hamiltonian = SparsePauliOp.from_sparse_list(sparse_list, 2)

	tt = hp.TruthTable(hamiltonian)
	assert tt.check_if_binary()
	assert tt.size == 2
	assert len(tt.truth_table.values()) == 2**2


def test_truth_table_operations() -> None:
	eq = hp.Equation(3)
	var0 = eq[0]
	var1 = eq[1]
	var2 = eq[2]
	assert isinstance(var0, hp.Variable)
	eq = (~var0 & var1) & var2
	eq = ~eq
	assert not eq.is_quadratic()
	tt = hp.TruthTable(eq)
	assert tt.count(0) == 1
	assert tt.count(1) == 2**3 - 1
	assert tt.count_min_value_solutions() == 1
	assert tt.get_solutions(0) == ['011']
	assert tt.get_min_value_solutions() == ['011']
	assert len(tt.get_solutions(1)) == 7
	assert tt.check_if_binary()
	assert tt[3] == 0
	assert tt[4] == 1


def test_with_hampy_object() -> None:
	equation = hp.Equation(4)
	equation = equation[0] & ~equation[1] & (equation[2] | equation[3])
	tt = hp.TruthTable(equation)
	assert tt.check_if_binary()
