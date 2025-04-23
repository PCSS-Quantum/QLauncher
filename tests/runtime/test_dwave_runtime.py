from quantum_launcher import QuantumLauncher
from quantum_launcher.routines.dwave_routines import DwaveSolver, SimulatedAnnealingBackend, TabuBackend
from quantum_launcher.problems import EC, JSSP, MaxCut, Raw, TSP
from pyqubo import Spin
TESTING_DIR = 'testing'


def _test_dwave_backends(problem):
    results = []
    solver = DwaveSolver(1)
    for backend in [
        SimulatedAnnealingBackend(),
        # TabuBackend() #Comment out because this thing is extremely slow...
    ]:
        launcher = QuantumLauncher(problem, solver, backend)

        inform = launcher.run()
        assert inform is not None
        results.append(inform)
    return results


def test_ec():
    """ Testing function for Exact Cover """
    pr = EC.from_preset(instance_name='micro')
    _test_dwave_backends(pr)


def test_jssp():
    """ Testing function for Job Shop Scheduling Problem """
    pr = JSSP.from_preset(instance_name='toy', optimization_problem=True)
    _test_dwave_backends(pr)


def test_maxcut():
    """ Testing function for Max Cut """
    pr = MaxCut.from_preset(instance_name='default')
    _test_dwave_backends(pr)


def test_raw():
    """ Testing function for Raw """
    qubits = [Spin(f"x{i}") for i in range(2)]
    H = 0
    H += 3 * qubits[0]
    H += 1 * qubits[1]
    H += -5 * qubits[0] * qubits[1]
    bqm = H.compile().to_bqm()
    pr = Raw(bqm)
    results = _test_dwave_backends(pr)
    for res in results:
        bitstring = res.best_bitstring
        assert bitstring in ['00', '01', '10', '11']


def test_tsp():
    """ Testing function for TSP """
    pr = TSP.generate_tsp_instance(3)  # Smaller sample size for testing
    _test_dwave_backends(pr)


def test_tabu_backend():
    qubits = [Spin(f"x{i}") for i in range(1)]
    H = 0
    H += -3 * qubits[0]
    bqm = H.compile().to_bqm()
    problem = Raw(bqm)
    solver = DwaveSolver(1)

    launcher = QuantumLauncher(problem, solver, TabuBackend())

    inform = launcher.run()
    assert inform is not None
