from quantum_launcher.hampy.object import HampyEquation, HampyVariable

from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Sampler
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.minimum_eigensolvers import QAOA
from pprint import pprint
from warnings import filterwarnings
filterwarnings('ignore')

def test_object():
    eq = HampyEquation(3)
    var0 = eq[0]
    var1 = eq[1]
    var2 = eq[2]
    assert isinstance(var0, HampyVariable)
    eq = (~var0 & var1) & var2
    eq = ~eq
    assert not eq.is_quadratic()
    hamiltonian = eq.to_sparse_pauli_op()
    assert isinstance(hamiltonian, SparsePauliOp)
    return hamiltonian

def test_run_qaoa():

    hamiltonian = test_object()
    sampler, optimizer = Sampler(), COBYLA()
    qaoa = QAOA(sampler, optimizer)
    result = qaoa.compute_minimum_eigenvalue(-hamiltonian)
    pprint(result.best_measurement)

def test_or_operation():

    eq = HampyEquation(2)
    var0 = eq.get_variable(0)
    var1 = eq.get_variable(1)
    new_eq = var0 | var1
    assert isinstance(new_eq, HampyEquation)

# test_run_qaoa()