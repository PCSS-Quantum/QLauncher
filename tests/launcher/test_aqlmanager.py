from quantum_launcher.launcher.aql import AQLManager, AQL
from quantum_launcher.problems import MaxCut, EC
from quantum_launcher.routines.dwave_routines import DwaveSolver, SimulatedAnnealingBackend
from quantum_launcher.routines.qiskit_routines import QAOA, QiskitBackend
from quantum_launcher.routines.orca_routines import BBS, OrcaBackend
import pytest


def test_runtime():
    return
    with AQLManager('test') as launcher:
        launcher.add(backend=SimulatedAnnealingBackend(),
                     algorithm=DwaveSolver(1), problem=EC.from_preset(onehot='exact', instance_name='micro'))
        launcher.add_algorithm(DwaveSolver(2))
        result = launcher.result

    assert len(result) == 2


def test_runtime_dwave():
    return
    with AQLManager('test') as launcher:
        launcher.add(backend=SimulatedAnnealingBackend(),
                     algorithm=DwaveSolver(1), problem=EC.from_preset(onehot='quadratic', instance_name='micro'))
        launcher.add_algorithm(DwaveSolver(2), times=2)
        launcher.add_problem(MaxCut.from_preset(instance_name='default'), times=3)
        result = launcher.result
        result_bitstring = launcher.result_bitstring

    assert len(result) == (2+1) * (3+1)
    assert len(result_bitstring) == (2+1) * (3+1)
    for x in result:
        assert x is not None
    for x in result_bitstring:
        assert isinstance(x, str)
        assert len(x) == 6 or len(x) == 2


def test_runtime_qiskit():
    return
    with AQLManager('test') as launcher:
        launcher.add(backend=QiskitBackend('local_simulator'),
                     algorithm=QAOA(2), problem=EC.from_preset(onehot='exact', instance_name='micro'))
        launcher.add_problem(MaxCut.from_preset(instance_name='default'), times=3)
        result = launcher.result
        result_bitstring = launcher.result_bitstring

    assert len(result) == (3+1)
    assert len(result_bitstring) == (3+1)
    for x in result:
        assert x is not None
    for x in result_bitstring:
        assert isinstance(x, str)
        assert len(x) == 6 or len(x) == 2


def test_runtime_orca():
    return
    # TODO Fix this test, it is not working as expected, it is not ending
    with AQLManager('test') as launcher:
        launcher.add(backend=OrcaBackend('local'),
                     algorithm=BBS(), problem=MaxCut.from_preset(instance_name='default'))
        launcher.add_problem(MaxCut.from_preset(instance_name='default'), times=3)
        result = launcher.result
        result_bitstring = launcher.result_bitstring

    assert len(result) == (3+1)
    assert len(result_bitstring) == (3+1)
    for x in result:
        assert x is not None
    for x in result_bitstring:
        assert isinstance(x, str)
        # assert len(x) == 10 or len(x) == 2


def test_AQL():
    aql = AQL()
    tasks = [
        (MaxCut.from_preset('default'), BBS(), OrcaBackend('local')),
        (MaxCut.from_preset('default'), BBS(), OrcaBackend('local')),
        (MaxCut.from_preset('default'), BBS(), OrcaBackend('local'))
    ]
    aql.add_task(tasks)
    aql.start()
    aql.wait_for_finish(timeout=10)

    res, bitres = aql.get_results()
    assert len(res) == len(bitres) == 3
