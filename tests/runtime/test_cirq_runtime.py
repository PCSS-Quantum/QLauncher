import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qlauncher import QLauncher
from qlauncher.base import Result
from qlauncher.base.models import Hamiltonian
from qlauncher.exceptions import DependencyError
from qlauncher.problems import MaxCut

try:
    import cirq

    from qlauncher.routines.cirq import CirqBackend

    CIRQ = True
except (DependencyError, ImportError):
    CIRQ = False
from qlauncher.routines.qiskit import QAOA, QiskitBackend


@pytest.mark.skipif(not CIRQ, reason='cirq not installed')
def test_cirq() -> None:
    problem = MaxCut.from_preset('default')
    algorithm = QAOA(p=2)
    backend = CirqBackend()
    launcher = QLauncher(problem, algorithm, backend)

    results = launcher.run()
    assert isinstance(results, Result)


@pytest.mark.skipif(not CIRQ, reason='cirq not installed')
def test_hamiltonian() -> None:
    """Testing function for Raw"""
    pr = Hamiltonian(SparsePauliOp.from_list([('ZZ', -1), ('ZI', 2), ('IZ', 2), ('II', -1)]))
    qaoa = QAOA()
    backend = CirqBackend()
    launcher = QLauncher(pr, qaoa, backend)

    results = launcher.run()
    assert results is not None
    bitstring = results.best_bitstring
    assert bitstring in ['00', '01', '10', '11']


@pytest.mark.skipif(not CIRQ, reason='cirq not installed')
def test_circuit() -> None:
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.measure_all()
    ql = QLauncher(qc, CirqBackend('local_simulator'))
    assert np.allclose(ql.run().distribution['0'], 0.5, atol=0.05)


@pytest.mark.skipif(not CIRQ, reason='cirq not installed')
def test_run_cirq_on_qiskit() -> None:
    q0 = cirq.GridQubit(0, 0)
    circuit = cirq.Circuit()
    circuit.append([cirq.H(q0)])
    circuit.append([cirq.measure(q0)])

    ql = QLauncher(circuit, QiskitBackend('local_simulator'))
    assert np.allclose(ql.run().distribution['0'], 0.5, atol=0.05)
