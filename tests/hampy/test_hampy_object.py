from warnings import filterwarnings

import pytest

import qlauncher.hampy as hp
from qlauncher import QLauncher
from qlauncher.base.models import Hamiltonian
from qlauncher.routines.qiskit import QAOA, QiskitBackend

filterwarnings('ignore')


def run_on_ql(hamiltonian: Hamiltonian) -> str:
    ql = QLauncher(hamiltonian, QAOA(), QiskitBackend('local_simulator'))
    return ql.run().best_bitstring[::-1]


def test_solving_hampy_eq() -> None:
    eq = hp.Equation(3)
    var0 = eq[0]
    var1 = eq[1]
    var2 = eq[2]
    assert isinstance(var0, hp.Variable)
    eq = (~var0 & var1) ^ var2
    eq = ~eq
    assert not eq.is_quadratic()
    hamiltonian = eq.to_sparse_pauli_op()
    bitstring = run_on_ql(Hamiltonian(hamiltonian))
    assert len(bitstring) == 3
    assert isinstance(bitstring, str)


def test_equation_equal() -> None:
    eq = hp.Equation(2)
    eq &= eq[0] & eq[1]
    eq2 = hp.Equation(2)
    eq2 &= (~eq2[0]) | (~eq2[1])
    assert eq == eq2
    with pytest.raises(TypeError):
        assert eq == 128
    one_var_equation = hp.Equation(1)
    assert one_var_equation[0].to_equation() == one_var_equation[0]


def test_equation_functions() -> None:
    eq = hp.Equation(3)
    eq = eq[0] | eq[1] & eq[2]

    assert eq.get_order() == 3


def test_or_operation() -> None:
    eq = hp.Equation(2)
    var0 = eq.get_variable(0)
    var1 = eq.get_variable(1)
    new_eq = var0 | var1
    assert isinstance(new_eq, hp.Equation)


def test_new_equation() -> None:
    equation = hp.Equation(5)
    equation = (equation[0] & ~equation[1] & (~equation[3] & equation[4])) | (equation[2] & equation[1])
    bitstring = run_on_ql(Hamiltonian((~equation).hamiltonian))
    assert len(bitstring) == 5


def test_one_in_n() -> None:
    equation = ~hp.one_in_n([0, 1, 2, 3, 4], 5)
    bitstring = run_on_ql(Hamiltonian(equation.hamiltonian))
    assert len(bitstring) == 5
    assert isinstance(bitstring, str)
    equation = hp.Equation(5)
    equation += hp.one_in_n(
        [
            equation[0],
            equation[1],
            equation[2],
            equation[3],
            equation[4],
        ],
        5,
        quadratic=True,
    )
    bitstring = run_on_ql(Hamiltonian(equation.hamiltonian))
    assert len(bitstring) == 5
    assert isinstance(bitstring, str)
