import numpy as np
import pytest
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit_machine_learning.kernels import BaseKernel

from qlauncher import QLauncher
from qlauncher.base import Result
from qlauncher.exceptions import DependencyError
from qlauncher.problems import TabularML

try:
	from qlauncher.routines.cirq import CirqBackend

	CIRQ = True
except DependencyError:
	CIRQ = False
from qlauncher.routines.qiskit import QiskitBackend, TrainQSVCKernel


def make_trainable_circ(n_qubits: int) -> tuple[QuantumCircuit, list[Parameter]]:
	circ = QuantumCircuit(n_qubits)

	trainable = [Parameter(f't{qb}') for qb in range(n_qubits)]

	for i, t in enumerate(trainable):
		circ.rx(t, i)

	for i in range(n_qubits):
		circ.ry(Parameter(f'{i}'), i)

	return circ, trainable


problem = TabularML(np.array([[5.2, 3.1], [11.3, 2.2]]), np.array([0, 1]))


def test_run_qiskit() -> None:
	alg = TrainQSVCKernel(*make_trainable_circ(2))
	backend = QiskitBackend('local_simulator')

	l = QLauncher(problem, alg, backend)
	r = l.run()
	assert isinstance(r, Result)
	assert isinstance(r.result, BaseKernel)


@pytest.mark.skipif(not CIRQ, reason='cirq not installed')
def test_run_no_trainable() -> None:
	circ = QuantumCircuit(2)

	for i in range(2):
		circ.ry(Parameter(f'{i}'), i)

	alg = TrainQSVCKernel(circ)
	backend = CirqBackend('local_simulator')

	l = QLauncher(problem, alg, backend)
	r = l.run()
	assert isinstance(r, Result)
	assert isinstance(r.result, BaseKernel)


@pytest.mark.skipif(not CIRQ, reason='cirq not installed')
def test_run_cirq() -> None:
	alg = TrainQSVCKernel(*make_trainable_circ(2))
	backend = CirqBackend('local_simulator')

	l = QLauncher(problem, alg, backend)
	r = l.run()
	assert isinstance(r, Result)
	assert isinstance(r.result, BaseKernel)
