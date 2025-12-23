import pytest

from qlauncher import QLauncher
from qlauncher.base import Result
from qlauncher.routines.pasqal import PasqalBackend, RydbergAnalogSolver
from tests.runtime.utils import ALL_PROBLEMS, PROBLEM_MAP
from tests.utils.problem import get_qubo


def test_rydberg_solver() -> None:
	"""Testing if Rydberg works with QUBO"""
	bbs = RydbergAnalogSolver()
	backend = PasqalBackend('local_simulator')
	launcher = QLauncher(get_qubo(), bbs, backend)

	inform = launcher.run()
	assert isinstance(inform, Result)


@pytest.mark.parametrize('problem_name', ALL_PROBLEMS)
def test_problem(problem_name: str) -> None:
	"""Testing problems using Rydberg"""
	bbs = RydbergAnalogSolver()
	backend = PasqalBackend('local_simulator')
	launcher = QLauncher(PROBLEM_MAP[problem_name], bbs, backend)

	inform = launcher.run()
	assert isinstance(inform, Result)
