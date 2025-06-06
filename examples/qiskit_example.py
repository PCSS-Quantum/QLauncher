""" Example of how Quantum Launcher works """
from quantum_launcher import QuantumLauncher, problems
from quantum_launcher.routines.qiskit import QiskitBackend, QAOA


def main():
    """ main """
    pr = problems.JSSP.from_preset('toy')
    alg = QAOA()
    backend = QiskitBackend('local_simulator')

    launcher = QuantumLauncher(pr, alg, backend)
    print(launcher.run())


if __name__ == '__main__':
    main()
