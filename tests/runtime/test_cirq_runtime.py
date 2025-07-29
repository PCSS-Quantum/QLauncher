from qiskit.quantum_info import SparsePauliOp

from qlauncher import QLauncher
from qlauncher.base import Result
from qlauncher.problems import MaxCut, Raw
from qlauncher.routines.qiskit import QAOA
from qlauncher.routines.cirq_routines import CirqBackend


def test_cirq():
    problem = MaxCut.from_preset('default')
    algorithm = QAOA(p=2)
    backend = CirqBackend()
    launcher = QLauncher(problem, algorithm, backend)

    results = launcher.run()
    assert isinstance(results, Result)


def test_raw():
    """ Testing function for Raw """
    hamiltonian = SparsePauliOp.from_list(
        [("ZZ", -1), ("ZI", 2), ("IZ", 2), ("II", -1)])
    pr = Raw(hamiltonian)
    qaoa = QAOA()
    backend = CirqBackend()
    launcher = QLauncher(pr, qaoa, backend)

    results = launcher.run()
    assert results is not None
    bitstring = results.best_bitstring
    assert bitstring in ['00', '01', '10', '11']
