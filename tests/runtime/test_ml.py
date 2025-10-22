import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit_machine_learning.kernels import BaseKernel

from qlauncher import QLauncher
from qlauncher.base import Result
from qlauncher.problems import TabularML
from qlauncher.routines.cirq import CirqBackend
from qlauncher.routines.qiskit import QiskitBackend, TrainQSVCKernel


def make_trainable_circ(n_qubits) -> tuple[QuantumCircuit, list[Parameter]]:
    circ = QuantumCircuit(n_qubits)

    trainable = [Parameter(f't{qb}') for qb in range(n_qubits)]

    for i, t in enumerate(trainable):
        circ.rx(t, i)

    for i in range(n_qubits):
        circ.ry(Parameter(f'{i}'), i)

    return circ, trainable


problem = TabularML(np.array([[5.2, 3.1], [11.3, 2.2]]), np.array([0, 1]))


def test_run_qiskit():
    alg = TrainQSVCKernel(*make_trainable_circ(2))
    backend = QiskitBackend('local_simulator')

    l = QLauncher(problem, alg, backend)
    r = l.run()
    assert isinstance(r, Result)
    assert isinstance(r.result, BaseKernel)


def test_run_no_trainable():
    circ = QuantumCircuit(2)

    for i in range(2):
        circ.ry(Parameter(f'{i}'), i)

    alg = TrainQSVCKernel(circ)
    backend = CirqBackend('local_simulator')

    l = QLauncher(problem, alg, backend)
    r = l.run()
    assert isinstance(r, Result)
    assert isinstance(r.result, BaseKernel)


def test_run_cirq():
    alg = TrainQSVCKernel(*make_trainable_circ(2))
    backend = CirqBackend('local_simulator')

    l = QLauncher(problem, alg, backend)
    r = l.run()
    assert isinstance(r, Result)
    assert isinstance(r.result, BaseKernel)
