from qiskit.quantum_info import SparsePauliOp

from quantum_launcher import QuantumLauncher
from quantum_launcher.base import Result
from quantum_launcher.routines.qiskit_routines import QAOA, QiskitBackend, FALQON
from quantum_launcher.routines.cirq_routines import CirqBackend
from quantum_launcher.problems import EC, JSSP, MaxCut, QATM, Raw, TSP
TESTING_DIR = 'testing'


def test_ec():
    """ Testing function for Exact Cover """
    pr = EC.from_preset(onehot='exact', instance_name='micro')
    qaoa = QAOA(p=3)
    backend = CirqBackend('local_simulator')
    launcher = QuantumLauncher(pr, qaoa, backend)

    # results = launcher.process(save_pickle=True, save_txt=True)
    results = launcher.run()
    assert isinstance(results, Result)


def test_qatm():
    """ Testing function for QATM """
    pr = QATM.from_file(instance_name='RCP_3.txt', instance_path='data/qatm/')
    qaoa = QAOA(p=3)
    backend = QiskitBackend('local_simulator')
    launcher = QuantumLauncher(pr, qaoa, backend)

    # results = launcher.process(save_pickle=True)
    results = launcher.run()
    assert isinstance(results, Result)


def test_jssp():
    """ Testing function for Job Shop Shedueling Problem """
    pr = JSSP.from_preset('toy', max_time=3, onehot='exact', optimization_problem=True)
    qaoa = QAOA(p=3)
    backend = QiskitBackend('local_simulator')
    launcher = QuantumLauncher(pr, qaoa, backend)

    # results = launcher.process(save_pickle=True)
    results = launcher.run()
    assert isinstance(results, Result)


def test_maxcut():
    """ Testing function for Max Cut """
    pr = MaxCut.from_preset(instance_name='default')
    qaoa = QAOA()
    backend = QiskitBackend('local_simulator')
    launcher = QuantumLauncher(pr, qaoa, backend)

    # results = launcher.process(save_pickle=True)
    results = launcher.run()
    assert isinstance(results, Result)


# def test_falqon():
#     """ Testing function for Falqon, using Exact Cover """
#     pr = EC('exact', instance_name='toy')
#     falqon = FALQON()
#     backend = QiskitBackend('local_simulator')
#     launcher = QuantumLauncher(pr, falqon, backend)

#     results = launcher.process(save_to_file=True)
#     assert results is not None


def test_raw():
    """ Testing function for Raw """
    hamiltonian = SparsePauliOp.from_list(
        [("ZZ", -1), ("ZI", 2), ("IZ", 2), ("II", -1)])
    pr = Raw(hamiltonian)
    qaoa = QAOA()
    backend = QiskitBackend('local_simulator')
    launcher = QuantumLauncher(pr, qaoa, backend)

    results = launcher.run()
    assert results is not None
    bitstring = results.best_bitstring
    assert bitstring in ['00', '01', '10', '11']


def test_tsp():
    """ Testing function for TSP """
    pr = TSP.generate_tsp_instance(3)  # Smaller sample size for testing
    qaoa = QAOA()
    backend = QiskitBackend('local_simulator')
    launcher = QuantumLauncher(pr, qaoa, backend)

    results = launcher.run()
    assert results is not None
    bitstring = results.best_bitstring
    assignments = [bitstring[i:i+3] for i in range(0, len(bitstring), 3)]
    assert len(assignments) == 3
    assert set(assignments) == set(['001', '010', '100'])
