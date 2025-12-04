import warnings
from pathlib import Path

import pytest

from qlauncher import QLauncher
from qlauncher.base.base import Result
from qlauncher.routines.qiskit import FALQON, QiskitBackend
from tests.utils.problem import get_hamiltonian


def prepare_launcher() -> QLauncher:
	algorithm = FALQON()
	backend = QiskitBackend('local_simulator')

	return QLauncher(get_hamiltonian(), algorithm, backend)


def test_params_are_bound() -> None:
	launcher = prepare_launcher()

	inform = launcher.run()

	assert isinstance(inform, Result)


def test_save(tmp_path: str) -> None:
	tmp = Path(tmp_path)
	launcher = prepare_launcher()
	with pytest.raises(ValueError):
		launcher.save(tmp / 'save.pckl', 'pickle')
	launcher.run()
	with pytest.raises(ValueError):
		launcher.save(tmp / 'save.pckl', 'pickel')  # type: ignore
	launcher.save(tmp / 'save.pkl', 'pickle')
	launcher.save(tmp / 'save.txt', 'txt')
	launcher.save(tmp / 'save.json', 'json')
