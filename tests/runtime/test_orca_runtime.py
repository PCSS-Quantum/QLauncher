from quantum_launcher import QuantumLauncher
from quantum_launcher.routines.orca_routines import BBS, OrcaBackend
from quantum_launcher.problems import EC, JSSP, MaxCut, QATM, Raw, TSP, GraphColoring
from quantum_launcher.base import Result
import numpy as np

TESTING_DIR = 'testing'


def test_ec():
    """ Testing function for Exact Cover """
    pr = EC.from_preset(onehot='quadratic', instance_name='toy')
    bbs = BBS()
    backend = OrcaBackend('local_simulator')
    launcher = QuantumLauncher(pr, bbs, backend)

    inform = launcher.run()
    assert isinstance(inform, Result)


def test_jssp():
    """ Testing function for Job Shop Shedueling Problem """
    pr = JSSP.from_preset(max_time=3, onehot='quadratic', instance_name='toy', optimization_problem=True)
    bbs = BBS()
    backend = OrcaBackend('local_simulator')
    launcher = QuantumLauncher(pr, bbs, backend)

    inform = launcher.run()
    assert isinstance(inform, Result)


def test_maxcut():
    """ Testing function for Max Cut """
    pr = MaxCut.from_preset(instance_name='default')
    bbs = BBS()
    backend = OrcaBackend('local_simulator')
    launcher = QuantumLauncher(pr, bbs, backend)

    inform = launcher.run()
    assert isinstance(inform, Result)


def test_raw():
    """ Testing function for Raw """
    qubo = np.array([[10, 1], [0, -10]]), 2
    pr = Raw(qubo)
    bbs = BBS()
    backend = OrcaBackend('local_simulator')
    launcher = QuantumLauncher(pr, bbs, backend)

    inform = launcher.run()
    assert isinstance(inform, Result)


def test_tsp():
    """ Testing function for TSP """
    pr = TSP.generate_tsp_instance(
        3, quadratic=True)  # Smaller sample size for testing
    bbs = BBS()
    backend = OrcaBackend('local_simulator')
    launcher = QuantumLauncher(pr, bbs, backend)

    inform = launcher.run()
    assert isinstance(inform, Result)


def test_graph_coloring():
    """Testing function for Graph Coloring"""
    gc = GraphColoring.from_preset("small")
    num_colors = gc.num_colors
    bbs = BBS()
    backend = OrcaBackend("local_simulator")
    launcher = QuantumLauncher(gc, bbs, backend)
    inform = launcher.run()
    assert isinstance(inform, Result)
    solution = inform.best_bitstring
    num_qubits = len(solution)
    assert num_qubits == gc.instance.number_of_nodes() * num_colors, "error in encoding, solution contains wrong number of qubits"
