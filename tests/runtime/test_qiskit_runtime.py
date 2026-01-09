import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeAthensV2  # Small 5qb backend for tests

from qlauncher import QLauncher
from qlauncher.base import Result
from qlauncher.routines.qiskit import FALQON, QAOA, QiskitBackend  # , AQTBackend
from qlauncher.routines.qiskit.algorithms.qiskit_native import VQE, Molecule
from qlauncher.utils import int_to_bitstring
from tests.runtime.utils import ALL_MITIGATION_STRATEGIES, ALL_PROBLEMS, MITIGATION_MAP, PROBLEM_MAP
from tests.utils.problem import get_hamiltonian


def test_int_to_bs() -> None:
	assert int_to_bitstring(5, 8) == '10100000'


def test_circuit() -> None:
	qc = QuantumCircuit(1)
	qc.h(0)
	qc.measure_all()
	ql = QLauncher(qc, QiskitBackend('local_simulator'))
	assert np.allclose(ql.run().distribution['0'], 0.5, atol=0.05)


def test_falqon() -> None:
	falqon = FALQON(max_reps=1)
	backend = QiskitBackend('local_simulator')
	launcher = QLauncher(get_hamiltonian(), falqon, backend)

	results = launcher.run()
	assert isinstance(results, Result)


def test_QAOA() -> None:
	qaoa = QAOA(p=1)
	backend = QiskitBackend('local_simulator')
	launcher = QLauncher(get_hamiltonian(), qaoa, backend)

	results = launcher.run()
	assert isinstance(results, Result)


def test_VQE() -> None:
	pr = Molecule.from_preset('H2')
	vqe = VQE()
	backend = QiskitBackend('local_simulator')
	launcher = QLauncher(pr, vqe, backend)

	results = launcher.run()
	assert isinstance(results, Result)


@pytest.mark.parametrize('mitigation_name', ALL_MITIGATION_STRATEGIES)
def test_mitigation(mitigation_name: str) -> None:
	backend = QiskitBackend(
		'backendv1v2', backendv1v2=FakeAthensV2(), error_mitigation_strategy=MITIGATION_MAP[mitigation_name], auto_transpile_level=0
	)
	qaoa = QAOA(p=1, max_evaluations=10)
	QLauncher(get_hamiltonian(), qaoa, backend).run()

	falqon = FALQON(max_reps=1)
	QLauncher(get_hamiltonian(), falqon, backend).run()


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
