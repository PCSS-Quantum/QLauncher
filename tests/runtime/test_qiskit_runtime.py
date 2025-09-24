import numpy as np

from qiskit.quantum_info import SparsePauliOp
from qlauncher import QLauncher
from qlauncher.base import Result
from qlauncher.routines.qiskit import QAOA, FALQON, QiskitBackend, AQTBackend
from qlauncher.routines.qiskit.algorithms.qiskit_native import int_to_bitstring
from qlauncher.routines.qiskit.algorithms.qiskit_native import Molecule, VQE
from qlauncher.problems import EC, JSSP, MaxCut, QATM, Raw, TSP, GraphColoring, Knapsack


def test_int_to_bs():
    assert int_to_bitstring(5, 8) == "10100000"


def test_falqon():
    pr = EC.from_preset(instance_name='micro')
    qaoa = FALQON(max_reps=1)
    backend = QiskitBackend('local_simulator')
    launcher = QLauncher(pr, qaoa, backend)

    results = launcher.run()
    assert isinstance(results, Result)


def test_QAOA():
    pr = EC.from_preset(instance_name='micro')
    qaoa = QAOA(p=1)
    backend = QiskitBackend('local_simulator')
    launcher = QLauncher(pr, qaoa, backend)

    results = launcher.run()
    assert isinstance(results, Result)


def test_VQE():
    pr = Molecule.from_preset('H2')
    vqe = VQE()
    backend = QiskitBackend('local_simulator')
    launcher = QLauncher(pr, vqe, backend)

    results = launcher.run()
    assert isinstance(results, Result)

#! We use FALQON for problem tests as it is very fast to execute


def test_ec():
    """ Testing function for Exact Cover """
    pr = EC.from_preset(instance_name='micro')
    qaoa = FALQON(max_reps=1)
    backend = QiskitBackend('local_simulator')
    launcher = QLauncher(pr, qaoa, backend)

    # results = launcher.process(save_pickle=True, save_txt=True)
    results = launcher.run()
    assert isinstance(results, Result)


def test_qatm():
    """ Testing function for QATM """
    pr = QATM.from_file(path='data/qatm/', instance_name='RCP_3.txt')
    qaoa = FALQON(max_reps=1)
    backend = QiskitBackend('local_simulator')
    launcher = QLauncher(pr, qaoa, backend)

    # results = launcher.process(save_pickle=True)
    results = launcher.run()
    assert isinstance(results, Result)


def test_jssp():
    """ Testing function for Job Shop Shedueling Problem """
    pr = JSSP.from_preset('default', optimization_problem=True)
    qaoa = FALQON(max_reps=1)
    backend = QiskitBackend('local_simulator')
    launcher = QLauncher(pr, qaoa, backend)

    # results = launcher.process(save_pickle=True)
    results = launcher.run()
    assert isinstance(results, Result)


def test_maxcut():
    """ Testing function for Max Cut """
    pr = MaxCut.from_preset(instance_name='default')
    qaoa = FALQON(max_reps=1)
    backend = QiskitBackend('local_simulator')
    launcher = QLauncher(pr, qaoa, backend)

    # results = launcher.process(save_pickle=True)
    results = launcher.run()
    assert isinstance(results, Result)


def test_raw():
    """ Testing function for Raw """
    hamiltonian = SparsePauliOp.from_list(
        [("ZZ", -1), ("ZI", 2), ("IZ", 2), ("II", -1)])
    pr = Raw(hamiltonian)
    qaoa = FALQON(max_reps=1)
    backend = QiskitBackend('local_simulator')
    launcher = QLauncher(pr, qaoa, backend)

    results = launcher.run()
    assert results is not None
    bitstring = results.best_bitstring
    assert bitstring in ['00', '01', '10', '11']


def test_tsp():
    """ Testing function for TSP """
    pr = TSP.generate_tsp_instance(3)  # Smaller sample size for testing
    qaoa = FALQON(max_reps=1)
    backend = QiskitBackend('local_simulator')
    launcher = QLauncher(pr, qaoa, backend)

    results = launcher.run()
    assert results is not None
    bitstring = results.best_bitstring
    assignments = [bitstring[i:i + 3] for i in range(0, len(bitstring), 3)]
    assert len(assignments) == 3


def test_graph_coloring():
    """Testing function for Graph Coloring"""
    gc = GraphColoring.from_preset("small")
    num_colors = gc.num_colors
    color_bit_length = int(np.ceil(np.log2(num_colors)))
    qaoa = FALQON(max_reps=1)
    backend = QiskitBackend("local_simulator")
    launcher = QLauncher(gc, qaoa, backend)
    inform = launcher.run()
    assert isinstance(inform, Result)
    bitstring = inform.best_bitstring
    num_qubits = len(bitstring)
    assert num_qubits == gc.instance.number_of_nodes(
    ) * color_bit_length, "error in encoding, solution contains wrong number of qubits"


def test_knapsack():
    """ Testing function for Knapsack problem """
    pr = Knapsack.from_preset("default")
    qaoa = FALQON(max_reps=1)
    backend = QiskitBackend("local_simulator")
    launcher = QLauncher(pr, qaoa, backend)

    results = launcher.run()
    assert isinstance(results, Result)
