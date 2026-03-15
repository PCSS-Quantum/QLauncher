import pytest

from qlauncher import QLauncher
from qlauncher.base import Result
from qlauncher.routines.orca import BBS, OrcaBackend
from tests.runtime.utils import ALL_PROBLEMS, PROBLEM_MAP
from tests.utils.problem import get_qubo


def test_bbs() -> None:
    """Testing if BBS works with QUBO"""
    bbs = BBS(updates=1)
    backend = OrcaBackend('local_simulator')
    launcher = QLauncher(get_qubo(), bbs, backend)

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
