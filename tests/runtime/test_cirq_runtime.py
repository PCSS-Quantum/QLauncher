import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qlauncher import QLauncher
from qlauncher.base import Result
from qlauncher.problems import MaxCut, Raw
from qlauncher.routines.cirq import CirqBackend
from qlauncher.routines.qiskit import QAOA


def test_cirq() -> None:
	problem = MaxCut.from_preset('default')
	algorithm = QAOA(p=2)
	backend = CirqBackend()
	launcher = QLauncher(problem, algorithm, backend)

	results = launcher.run()
	assert isinstance(results, Result)


def test_raw() -> None:
	"""Testing function for Raw"""
	hamiltonian = SparsePauliOp.from_list([('ZZ', -1), ('ZI', 2), ('IZ', 2), ('II', -1)])
	pr = Raw(hamiltonian)
	qaoa = QAOA()
	backend = CirqBackend()
	launcher = QLauncher(pr, qaoa, backend)

	results = launcher.run()
	assert results is not None
	bitstring = results.best_bitstring
	assert bitstring in ['00', '01', '10', '11']


def test_circuit() -> None:
	qc = QuantumCircuit(1)
	qc.h(0)
	qc.measure_all()
	ql = QLauncher(qc, CirqBackend('local_simulator'))
	assert np.allclose(ql.run().distribution['0'], 0.5, atol=0.05)
