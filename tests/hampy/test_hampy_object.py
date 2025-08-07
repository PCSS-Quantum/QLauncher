from pprint import pprint
from warnings import filterwarnings

from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Sampler
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.minimum_eigensolvers import QAOA

from qlauncher.hampy.object import Equation, Variable
from qlauncher.hampy.equations import one_in_n
filterwarnings('ignore')


def get_hamiltonian():
    eq = Equation(3)
    var0 = eq[0]
    var1 = eq[1]
    var2 = eq[2]
    assert isinstance(var0, Variable)
    eq = (~var0 & var1) & var2
    eq = ~eq
    assert not eq.is_quadratic()
    hamiltonian = eq.to_sparse_pauli_op()
    assert isinstance(hamiltonian, SparsePauliOp)
    return hamiltonian


def test_run_qaoa():

    hamiltonian = get_hamiltonian()
    sampler, optimizer = Sampler(), COBYLA()
    qaoa = QAOA(sampler, optimizer)
    result = qaoa.compute_minimum_eigenvalue(-hamiltonian)
    pprint(result.best_measurement)


def test_or_operation():

    eq = Equation(2)
    var0 = eq.get_variable(0)
    var1 = eq.get_variable(1)
    new_eq = var0 | var1
    assert isinstance(new_eq, Equation)


def test_one_in_n():
    equation = ~one_in_n([0, 1, 2, 3, 4], 5)
    sampler, optimizer = Sampler(), COBYLA()
    qaoa = QAOA(sampler, optimizer)
    result = qaoa.compute_minimum_eigenvalue(equation.hamiltonian)
    bitstring: str = result.best_measurement['bitstring']
    assert bitstring.count('1') == 1


def test_new_equation():
    equation = Equation(5)
    equation = (equation[0] & ~equation[1] & (~equation[3] & equation[4])) | (equation[2] & equation[1])
    hamiltonian = (~equation).hamiltonian
    sampler, optimizer = Sampler(), COBYLA()
    qaoa = QAOA(sampler, optimizer)
    result = qaoa.compute_minimum_eigenvalue(hamiltonian)
    assert len(result.best_measurement['bitstring']) == 5
