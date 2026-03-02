import pytest
from pyqubo import Spin

from qlauncher import QLauncher
from qlauncher.base import Result
from qlauncher.base.problem_like import BQM
from qlauncher.routines.dwave import TabuBackend
from qlauncher.routines.dwave.algorithms import SimulatedAnnealing, SteepestDescent, Tabu
from tests.runtime.utils import ALL_PROBLEMS, PROBLEM_MAP

TESTING_DIR = 'testing'
ALGORITHM_MAP = {
	'SimulatedAnnealing': SimulatedAnnealing(num_reads=10),
	'SteepestDescent': SteepestDescent(num_reads=10),
	'Tabu': Tabu(num_reads=10),
}
ALL_ALGORITHMS = list(ALGORITHM_MAP.keys())


def _get_bqm() -> BQM:
	qubits = [Spin(f'x{i}') for i in range(2)]
	H = 3 * qubits[0] - 5 * qubits[0] * qubits[1]
	model = H.compile()

	return BQM(model)


@pytest.mark.parametrize('algorithm_name', ALL_ALGORITHMS)
def test_algorithms(algorithm_name: str) -> None:
	"""Testing D-wave Algorithms"""
	launcher = QLauncher(_get_bqm(), ALGORITHM_MAP[algorithm_name], TabuBackend())

	inform = launcher.run()
	assert isinstance(inform, Result)


@pytest.mark.parametrize('problem_name', ALL_PROBLEMS)
def test_problem(problem_name: str) -> None:
	"""Testing problems using Simulated Annealing"""
	backend = TabuBackend()
	launcher = QLauncher(PROBLEM_MAP[problem_name], ALGORITHM_MAP['SimulatedAnnealing'], backend)
	inform = launcher.run()
	assert isinstance(inform, Result)
