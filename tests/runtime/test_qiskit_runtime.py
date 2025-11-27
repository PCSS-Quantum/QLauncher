import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qlauncher import QLauncher
from qlauncher.base import Result
from qlauncher.base.problem_like import Hamiltonian
from qlauncher.routines.qiskit import FALQON, QAOA, QiskitBackend  # , AQTBackend
from qlauncher.routines.qiskit.algorithms.qiskit_native import VQE, Molecule
from qlauncher.utils import int_to_bitstring
from tests.runtime.utils import ALL_PROBLEMS, PROBLEM_MAP


def _get_hamiltonian() -> Hamiltonian:
	return Hamiltonian(SparsePauliOp.from_list([('ZZ', -1), ('ZI', 2), ('IZ', 2), ('II', -1)]))


def test_int_to_bs() -> None:
	assert int_to_bitstring(5, 8) == '10100000'


def test_circuit() -> None:
	qc = QuantumCircuit(1)
	qc.h(0)
	qc.measure_all()
	ql = QLauncher(qc, QiskitBackend('local_simulator'))
	assert np.allclose(ql.run().distribution['0'], 0.5, atol=0.05)


def test_falqon() -> None:
	qaoa = FALQON(max_reps=1)
	backend = QiskitBackend('local_simulator')
	launcher = QLauncher(_get_hamiltonian(), qaoa, backend)

	results = launcher.run()
	assert isinstance(results, Result)


def test_QAOA() -> None:
	qaoa = QAOA(p=1)
	backend = QiskitBackend('local_simulator')
	launcher = QLauncher(_get_hamiltonian(), qaoa, backend)

	results = launcher.run()
	assert isinstance(results, Result)


def test_VQE() -> None:
	pr = Molecule.from_preset('H2')
	vqe = VQE()
	backend = QiskitBackend('local_simulator')
	launcher = QLauncher(pr, vqe, backend)

	results = launcher.run()
	assert isinstance(results, Result)


#! We use FALQON for problem tests as it is very fast to execute
@pytest.mark.parametrize('problem_name', ALL_PROBLEMS)
def test_problems(problem_name: str) -> None:
	"""Testing function for Exact Cover"""
	problem = PROBLEM_MAP[problem_name]
	qaoa = FALQON(max_reps=1)
	backend = QiskitBackend('local_simulator')
	launcher = QLauncher(problem, qaoa, backend)

	# results = launcher.process(save_pickle=True, save_txt=True)
	results = launcher.run()
	assert isinstance(results, Result)
