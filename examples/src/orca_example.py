"""QLauncher for Orca"""

from qlauncher import QLauncher
from qlauncher.problems import MaxCut
from qlauncher.routines.orca import OrcaBackend, BBS


def main():
	"""main"""
	problem = MaxCut.from_preset(instance_name='default')
	alg = BBS()
	backend = OrcaBackend('local')
	launcher = QLauncher(problem, alg, backend)
	result = launcher.run()
	print(result)


if __name__ == '__main__':
	main()
