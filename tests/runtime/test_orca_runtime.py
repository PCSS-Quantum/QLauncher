import numpy as np

from qlauncher import QLauncher
from qlauncher.base import Result
from qlauncher.base.problem_like import QUBO
from qlauncher.problems import EC, JSSP, TSP, GraphColoring, Knapsack, MaxCut
from qlauncher.routines.orca import BBS, OrcaBackend


def test_ec() -> None:
	"""Testing function for Exact Cover"""
	pr = EC.from_preset(instance_name='micro')
	bbs = BBS(updates=1)
	backend = OrcaBackend('local_simulator')
	launcher = QLauncher(pr, bbs, backend)

	inform = launcher.run()
	assert isinstance(inform, Result)


def test_jssp() -> None:
	"""Testing function for Job Shop Shedueling Problem"""
	pr = JSSP.from_preset(instance_name='default', optimization_problem=True)
	bbs = BBS(updates=1)
	backend = OrcaBackend('local_simulator')
	launcher = QLauncher(pr, bbs, backend)

	inform = launcher.run()
	assert isinstance(inform, Result)


def test_maxcut() -> None:
	"""Testing function for Max Cut"""
	pr = MaxCut.from_preset(instance_name='default')
	bbs = BBS(updates=1)
	backend = OrcaBackend('local_simulator')
	launcher = QLauncher(pr, bbs, backend)

	inform = launcher.run()
	assert isinstance(inform, Result)


def test_qubo() -> None:
	"""Testing function for QUBO"""
	launcher = QLauncher(
		QUBO(np.array([[10, 1], [0, -10]]), 2),
		BBS(updates=1),
		OrcaBackend('local_simulator'),
	)

	inform = launcher.run()
	assert isinstance(inform, Result)


def test_tsp() -> None:
	"""Testing function for TSP"""
	pr = TSP.generate_tsp_instance(3)  # Smaller sample size for testing
	bbs = BBS(updates=1)
	backend = OrcaBackend('local_simulator')
	launcher = QLauncher(pr, bbs, backend)

	inform = launcher.run()

	assert isinstance(inform, Result)


def test_graph_coloring() -> None:
	"""Testing function for Graph Coloring"""
	gc = GraphColoring.from_preset('small')
	num_colors = gc.num_colors
	bbs = BBS(updates=1)
	backend = OrcaBackend('local_simulator')
	launcher = QLauncher(gc, bbs, backend)
	inform = launcher.run()
	assert isinstance(inform, Result)
	solution = inform.best_bitstring
	num_qubits = len(solution)
	assert num_qubits == gc.instance.number_of_nodes() * num_colors, 'error in encoding, solution contains wrong number of qubits'


def test_knapsack() -> None:
	"""Testing function for Knapsack problem"""
	pr = Knapsack.from_preset(instance_name='default')
	bbs = BBS(updates=1)
	backend = OrcaBackend('local_simulator')
	launcher = QLauncher(pr, bbs, backend)

	inform = launcher.run()
	assert isinstance(inform, Result)
