""" Test for raw problems """
import numpy as np
from qiskit.quantum_info import SparsePauliOp

from qlauncher import QLauncher, Result
from qlauncher.problems import Raw
from qlauncher.routines.qiskit import QiskitBackend, QAOA


def test_auto_assigning():
    """ Tests if 2 of the same type raw's have the same class
    while also different types have different class. """
    hamiltonian = SparsePauliOp.from_list([('IZ', 2), ('ZI', 2)])
    hamiltonian2 = SparsePauliOp.from_list([('IZ', 1), ('ZZ', .5)])
    qubo = (np.array([[0, 1], [0, 1]]), 1)
    raw = Raw(hamiltonian)
    raw2 = Raw(hamiltonian2)
    raw_qubo = Raw(qubo)
    assert raw.__class__ is raw2.__class__
    assert raw.__class__ is not raw_qubo.__class__


def test_auto_formatting_hamiltonian():
    """ Test if Raw assigning works properly """
    hamiltonian = SparsePauliOp.from_list([('IZ', 2), ('ZI', 2)])

    ql = QLauncher(hamiltonian, QAOA(p=1), backend=QiskitBackend('local_simulator'))
    assert isinstance(ql.problem, Raw)
    res = ql.run()
    assert isinstance(res, Result)
