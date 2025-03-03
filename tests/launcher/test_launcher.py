from quantum_launcher import QuantumLauncher
from quantum_launcher.routines.orca_routines import BBS, OrcaBackend
from quantum_launcher.routines.qiskit_routines import QAOA, QiskitBackend
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

    inform = launcher.run(onehot="quadratic")  # This will fail if the parameter is not set by run()


def test_params_are_reset():
    launcher = prepare_launcher()

    inform = launcher.run(onehot="quadratic")

    with pytest.raises(Exception):
        inform = launcher.run()  # Should not keep past params


def test_unused_params_raise_warning():
    launcher = prepare_launcher()

    with pytest.warns(Warning):
        inform = launcher.run(onehot="quadratic", unused=123)
