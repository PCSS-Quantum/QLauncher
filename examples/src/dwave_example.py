"""QLauncher for Orca"""

from qlauncher import *
from qlauncher import problems
from qlauncher.routines.dwave import DwaveSolver, SimulatedAnnealingBackend


def main():
    """main"""
    problem = problems.MaxCut(instance_name='default')
    alg = DwaveSolver(1)
    backend = SimulatedAnnealingBackend('local')
    launcher = QLauncher(problem, alg, backend)
    res = launcher.run()
    print(alg.get_bitstring(res))


if __name__ == '__main__':
    main()
