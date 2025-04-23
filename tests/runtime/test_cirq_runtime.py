from qiskit.quantum_info import SparsePauliOp

from quantum_launcher import QuantumLauncher
from quantum_launcher.base import Result
from quantum_launcher.problems import MaxCut, Hamiltonian
from quantum_launcher.routines.qiskit_routines import QAOA
from quantum_launcher.routines.cirq_routines import CirqBackend


def test_cirq():
    problem = MaxCut.from_preset('default')
    algorithm = QAOA(p=2)
    backend = CirqBackend()
    launcher = QuantumLauncher(problem, algorithm, backend)

    results = launcher.run()
    assert isinstance(results, Result)


def test_raw():
    """ Testing function for Raw """
    hamiltonian = SparsePauliOp.from_list(
        [("ZZ", -1), ("ZI", 2), ("IZ", 2), ("II", -1)])
    pr = Hamiltonian(hamiltonian)
    qaoa = QAOA()
    backend = CirqBackend()
    launcher = QuantumLauncher(pr, qaoa, backend)

    results = launcher.run()
    assert results is not None
    bitstring = results.best_bitstring
    assert bitstring in ['00', '01', '10', '11']
