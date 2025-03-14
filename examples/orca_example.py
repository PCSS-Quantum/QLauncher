""" Quantum Launcher for Orca """
from quantum_launcher import QuantumLauncher
from quantum_launcher.problems import MaxCut
from quantum_launcher.routines.orca_routines import OrcaBackend, BBS


def main():
    """ main """
    problem = MaxCut.from_preset(instance_name='default')
    alg = BBS()
    backend = OrcaBackend('local')
    launcher = QuantumLauncher(problem, alg, backend)
    result = launcher.run()
    print(result)


if __name__ == '__main__':
    main()
