import numpy as np
import pytest

from qlauncher import QLauncher
from qlauncher.base import Result
from qlauncher.base.problem_like import QUBO
from qlauncher.routines.orca import BBS, OrcaBackend
from tests.runtime.utils import ALL_PROBLEMS, PROBLEM_MAP


def _get_qubo() -> QUBO:
	return QUBO(np.array([[1, 0], [-10, 1]]), 2)


def test_bbs() -> None:
	"""Testing if BBS works with QUBO"""
	bbs = BBS(updates=1)
	backend = OrcaBackend('local_simulator')
	launcher = QLauncher(_get_qubo(), bbs, backend)

	inform = launcher.run()
	assert isinstance(inform, Result)


@pytest.mark.parametrize('problem_name', ALL_PROBLEMS)
def test_problem(problem_name: str) -> None:
	"""Testing problems using BBS"""
	bbs = BBS(updates=1)
	backend = OrcaBackend('local_simulator')
	launcher = QLauncher(PROBLEM_MAP[problem_name], bbs, backend)

	inform = launcher.run()
	assert isinstance(inform, Result)
