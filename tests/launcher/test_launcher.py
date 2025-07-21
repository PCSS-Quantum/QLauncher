import os
from qlauncher import QuantumLauncher
from qlauncher.routines.qiskit_routines import FALQON, QiskitBackend
from qlauncher.base.base import Result
from qlauncher.problems import TSP
import warnings
import pytest


def prepare_launcher():
    problem = TSP.generate_tsp_instance(3)

    algorithm = FALQON()
    backend = QiskitBackend('local_simulator')

    launcher = QuantumLauncher(problem, algorithm, backend)

    return launcher


def test_params_are_bound():
    launcher = prepare_launcher()

    inform = launcher.run()

    assert isinstance(inform, Result)


def test_unused_params_raise_warning():
    launcher = prepare_launcher()

    with pytest.warns(Warning):
        inform = launcher.run(unused=123)

    assert isinstance(inform, Result)


def test_save(tmp_path):
    launcher = prepare_launcher()
    with pytest.raises(ValueError):
        launcher.save(os.path.join(tmp_path, "save.pckl"), 'pickle')
    launcher.run()
    with pytest.raises(ValueError):
        launcher.save(os.path.join(tmp_path, "save.pckl"), 'pickel')
    launcher.save(os.path.join(tmp_path, "save.pckl"), 'pickle')
    launcher.save(os.path.join(tmp_path, "save.txt"), 'txt')
    launcher.save(os.path.join(tmp_path, "save.json"), 'json')


@pytest.mark.skip('Currently getting qiskit deprecation warning')
def test_override_params_raise_warning():
    launcher = prepare_launcher()

    # overriding onehot='quadratic' required by hamiltonian_to_qubo
    with pytest.warns(Warning):
        inform = launcher.run(onehot='exact')

    # test if setting other params generates no warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        inform = launcher.run(constraints_weight=10)

    assert isinstance(inform, Result)
