from quantum_launcher import QuantumLauncher
from quantum_launcher.routines.orca_routines import BBS, OrcaBackend
from quantum_launcher.problems import TSP


def test_param_binding():
    problem = TSP.generate_tsp_instance(3)
    assert problem.quadratic == False

    algorithm = BBS()
    backend = OrcaBackend('local_simulator')

    launcher = QuantumLauncher(problem, algorithm, backend)

    inform = launcher.run(problem__quadratic=True)  # This will fail if the parameter is not set by run()

    assert problem.quadratic == False  # Check if original value is restored
