from quantum_launcher import QuantumLauncher
from quantum_launcher.routines.orca_routines import BBS, OrcaBackend
from quantum_launcher.routines.qiskit_routines import QAOA, QiskitBackend
from quantum_launcher.problems import TSP

from quantum_launcher.base.adapter_structure import Formatter


def test_adapter_binding():
    problem = TSP.generate_tsp_instance(3)

    algorithm = BBS()
    backend = OrcaBackend('local_simulator')

    launcher = QuantumLauncher(problem, algorithm, backend)

    inform = launcher.run(onehot="quadratic")  # This will fail if the parameter is not set by run()


def test_formatter_binding():
    problem = TSP.generate_tsp_instance(3)

    algorithm = QAOA()
    backend = QiskitBackend('local_simulator')

    launcher = QuantumLauncher(problem, algorithm, backend)
    
    assert isinstance(launcher.formatter, Formatter)

    inform = launcher.run(onehot="quadratic") 