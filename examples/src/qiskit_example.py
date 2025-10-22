"""Example of how QLauncher works"""

from qlauncher import QLauncher, problems
from qlauncher.routines.qiskit import QAOA, IBMBackend


def main():
    """main"""
    pr = problems.JSSP.from_preset('toy')
    alg = QAOA()
    backend = IBMBackend('local_simulator')

    launcher = QLauncher(pr, alg, backend)
    print(launcher.run())


if __name__ == '__main__':
    main()
