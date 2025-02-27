from quantum_launcher import QuantumLauncher
from quantum_launcher.problems import MaxCut
from quantum_launcher.routines.qiskit_routines import QAOA
from quantum_launcher.routines.cirq_routines import CirqBackend


def test_cirq():
    problem = MaxCut.from_preset('default')
    algorithm = QAOA(p=2)
    backend = CirqBackend()
    ql = QuantumLauncher(problem, algorithm, backend)

    result = ql.run()
