""" Quantum Launcher for Orca """
from qlauncher import QuantumLauncher
from qlauncher.problems import MaxCut
from qlauncher.routines.orca_routines import OrcaBackend, BBS


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
