from qlauncher import QLauncher
from qlauncher.base import Result
from qlauncher.routines.dwave import DwaveSolver, SimulatedAnnealingBackend, TabuBackend, SteepestDescentBackend
from qlauncher.problems import EC, JSSP, MaxCut, Raw, TSP, Knapsack
from pyqubo import Spin

TESTING_DIR = 'testing'


def _test_with_backend(problem, solver, backend):
	launcher = QLauncher(problem, solver, backend)

	inform = launcher.run()
	assert isinstance(inform, Result)
	return inform


def _test_with_simulated_annealing(problem, solver):
	return _test_with_backend(problem, solver, SimulatedAnnealingBackend())


def _test_with_steepest_descent(problem, solver):
	return _test_with_backend(problem, solver, SteepestDescentBackend())


def _test_with_tabu(problem, solver):
	return _test_with_backend(problem, solver, TabuBackend())


# Repeated code for verboseness of errors
def test_ec():
	"""Testing function for Exact Cover"""
	problem = EC.from_preset(instance_name='micro')
	solver = DwaveSolver(1, num_reads=10)
	_test_with_simulated_annealing(problem, solver)
	_test_with_steepest_descent(problem, solver)


def test_jssp():
	"""Testing function for Job Shop Scheduling Problem"""
	problem = JSSP.from_preset(instance_name='default', optimization_problem=True)
	solver = DwaveSolver(1, num_reads=10)
	_test_with_simulated_annealing(problem, solver)
	_test_with_steepest_descent(problem, solver)


def test_maxcut():
	"""Testing function for Max Cut"""
	problem = MaxCut.from_preset(instance_name='default')
	solver = DwaveSolver(1, num_reads=10)
	_test_with_simulated_annealing(problem, solver)
	_test_with_steepest_descent(problem, solver)


def test_raw():
	"""Testing function for Raw"""
	qubits = [Spin(f'x{i}') for i in range(2)]
	H = 0
	H += 3 * qubits[0]
	H += 1 * qubits[1]
	H += -5 * qubits[0] * qubits[1]
	bqm = H.compile().to_bqm()
	problem = Raw(bqm)
	solver = DwaveSolver(1, num_reads=10)
	results = [_test_with_simulated_annealing(problem, solver), _test_with_steepest_descent(problem, solver)]
	for res in results:
		bitstring = res.best_bitstring
		assert bitstring in ['00', '01', '10', '11']


def test_tsp():
	"""Testing function for TSP"""
	problem = TSP.generate_tsp_instance(3)  # Smaller sample size for testing
	solver = DwaveSolver(1, num_reads=10)
	_test_with_simulated_annealing(problem, solver)
	_test_with_steepest_descent(problem, solver)


def test_tabu_backend():
	qubits = [Spin(f'x{i}') for i in range(1)]
	H = 0
	H += -3 * qubits[0]
	bqm = H.compile().to_bqm()
	problem = Raw(bqm)
	solver = DwaveSolver(1, timeout=0.1, num_restarts=1)

	_test_with_tabu(problem, solver)


def test_knapsack():
	"""Testing function for Knapsack problem"""
	problem = Knapsack.from_preset(instance_name='default')
	solver = DwaveSolver(1, num_reads=10)
	_test_with_simulated_annealing(problem, solver)
	_test_with_steepest_descent(problem, solver)
