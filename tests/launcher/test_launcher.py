from pathlib import Path
from typing import Any

import pytest

from qiskit.quantum_info import SparsePauliOp

from qlauncher import QLauncher
from qlauncher.base.base import Result, Problem
from qlauncher.base.models import Hamiltonian
from qlauncher.routines.qiskit import FALQON, QiskitBackend
from tests.utils.problem import get_hamiltonian


class DummyProblem(Problem):
	def to_hamiltonian(
		self,
		v=None,
	) -> Hamiltonian:
		if v is None:
			raise ValueError('Param not set!')
		return Hamiltonian(SparsePauliOp.from_list([('Z', 1.0)]))


def prepare_launcher() -> QLauncher:
	algorithm = FALQON()
	backend = QiskitBackend('local_simulator')

	return QLauncher(get_hamiltonian(), algorithm, backend)


def test_params_are_bound() -> None:
	launcher = QLauncher(DummyProblem(None), FALQON(), QiskitBackend('local_simulator'))

	inform = launcher.run(v=2)

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
