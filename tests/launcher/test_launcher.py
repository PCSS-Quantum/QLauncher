from quantum_launcher import QuantumLauncher
from quantum_launcher.routines.orca_routines import BBS, OrcaBackend
from quantum_launcher.base.base import Result
from quantum_launcher.problems import TSP
import warnings
import pytest


def prepare_launcher():
    problem = TSP.generate_tsp_instance(3)

    algorithm = BBS()
    backend = OrcaBackend('local_simulator')

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
