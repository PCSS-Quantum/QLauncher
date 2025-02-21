import numpy as np
from quantum_launcher import QuantumLauncher
from quantum_launcher.base import Result
from quantum_launcher.routines.qiskit_routines import QAOA, QiskitBackend, FALQON
from quantum_launcher.problems import EC, JSSP, MaxCut, QATM, Raw, TSP, GraphColoring
from qiskit.quantum_info import SparsePauliOp

TESTING_DIR = "testing"


def test_ec():
    """ Testing function for Exact Cover """
    pr = EC.from_preset(onehot='exact', instance_name='micro')
    qaoa = QAOA(p=3)
    backend = QiskitBackend('local_simulator')
    launcher = QuantumLauncher(pr, qaoa, backend)

    # inform = launcher.process(save_pickle=True, save_txt=True)
    inform = launcher.run()
    assert isinstance(inform, Result)


def test_qatm():
    """ Testing function for QATM """
    pr = QATM.from_file(instance_name='RCP_3.txt', instance_path='data/qatm/')
    qaoa = QAOA(p=3)
    backend = QiskitBackend('local_simulator')
    launcher = QuantumLauncher(pr, qaoa, backend)

    # inform = launcher.process(save_pickle=True)
    inform = launcher.run()
    assert isinstance(inform, Result)


def test_jssp():
    """ Testing function for Job Shop Shedueling Problem """
    pr = JSSP.from_preset('toy', max_time=3, onehot='exact', optimization_problem=True)
    qaoa = QAOA(p=3)
    backend = QiskitBackend('local_simulator')
    launcher = QuantumLauncher(pr, qaoa, backend)

    # inform = launcher.process(save_pickle=True)
    inform = launcher.run()
    assert isinstance(inform, Result)


def test_maxcut():
    """ Testing function for Max Cut """
    pr = MaxCut.from_preset(instance_name='default')
    qaoa = QAOA()
    backend = QiskitBackend('local_simulator')
    launcher = QuantumLauncher(pr, qaoa, backend)

    # inform = launcher.process(save_pickle=True)
    inform = launcher.run()
    assert isinstance(inform, Result)


# def test_falqon():
#     """ Testing function for Falqon, using Exact Cover """
#     pr = EC('exact', instance_name='toy')
#     falqon = FALQON()
#     backend = QiskitBackend('local_simulator')
#     launcher = QuantumLauncher(pr, falqon, backend)

#     inform = launcher.process(save_to_file=True)
#     assert inform is not None


def test_raw():
    """ Testing function for Raw """
    hamiltonian = SparsePauliOp.from_list(
        [("ZZ", -1), ("ZI", 2), ("IZ", 2), ("II", -1)])
    pr = Raw(hamiltonian)
    qaoa = QAOA()
    backend = QiskitBackend('local_simulator')
    launcher = QuantumLauncher(pr, qaoa, backend)

    inform = launcher.run()
    assert inform is not None
    bitstring = inform.best_bitstring
    assert bitstring in ['00', '01', '10', '11']


def test_tsp():
    """ Testing function for TSP """
    pr = TSP.generate_tsp_instance(3)  # Smaller sample size for testing
    qaoa = QAOA()
    backend = QiskitBackend('local_simulator')
    launcher = QuantumLauncher(pr, qaoa, backend)

    inform = launcher.run()
    assert inform is not None
    bitstring = inform.best_bitstring
    assignments = [bitstring[i:i+3] for i in range(0, len(bitstring), 3)]
    assert len(assignments) == 3
    assert set(assignments) == set(['001', '010', '100'])


def test_graph_coloring():
    """Testing function for Graph Coloring"""
    gc = GraphColoring.from_preset("small")
    num_colors = gc.num_colors
    color_bit_length = int(np.ceil(np.log2(num_colors)))
    qaoa = QAOA()
    backend = QiskitBackend("local_simulator")
    launcher = QuantumLauncher(gc, qaoa, backend)
    inform = launcher.run()
    assert isinstance(inform, Result)
    bitstring = inform.best_bitstring
    num_qubits = len(bitstring)
    assert num_qubits == gc.instance.number_of_nodes() * color_bit_length
    solution = bitstring[::-1]
    proper_solution = []
    for i in range(0, num_qubits, color_bit_length):
        node_bit_string = solution[i: i + color_bit_length]
        color_num = int(node_bit_string, 2)
        proper_solution.append(color_num)
    assert len(proper_solution) == gc.instance.number_of_nodes()
    assert all([x in set(range(num_colors)) for x in proper_solution])
    assert len(set(proper_solution[:-1])) == len(proper_solution[:-1])
