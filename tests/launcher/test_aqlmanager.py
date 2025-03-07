from concurrent.futures import ThreadPoolExecutor

from quantum_launcher.launcher.aql import AQL, AQLTask
from quantum_launcher.launcher.qlauncher import QuantumLauncher
from quantum_launcher.problems import MaxCut, EC
from quantum_launcher.routines.dwave_routines import DwaveSolver, SimulatedAnnealingBackend
from quantum_launcher.routines.qiskit_routines import QAOA, QiskitBackend
from quantum_launcher.routines.orca_routines import BBS, OrcaBackend
import pytest

import numpy as np


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


def test_AQL_individual_tasks():
    aql = AQL()

    aql.add_task((MaxCut.from_preset('default'), QAOA(), QiskitBackend('local_simulator')))
    aql.add_task((MaxCut.from_preset('default'), BBS(), OrcaBackend('local_simulator')))
    aql.add_task((MaxCut.from_preset('default'), DwaveSolver(), SimulatedAnnealingBackend('local_simulator')))

    aql.start()
    aql.wait_for_finish(timeout=10)

    res, bitres = aql.get_results()
    assert len(res) == len(bitres) == 3


def test_AQL_chained_tasks():
    """
    Check that tasks in a chain execute one after another.
    """
    # TODO change to DwaveBackend with minimal work to make this faster
    launchers = [QuantumLauncher(MaxCut.from_preset('default'), QAOA(), QiskitBackend('local_simulator')) for _ in range(5)]

    order = []

    aql = AQL()
    aql.add_task_chain(launchers)
    wanted = aql.tasks.copy()
    for t in aql.tasks:
        t.callbacks.append(order.append)

    np.random.shuffle(aql.tasks)  # Shuffle the order of starting tasks, if dependencies work correctly, this should make no difference.
    aql.start()
    aql.wait_for_finish(10)

    assert len(aql.get_results()[0]) == 5
    assert order == wanted


def test_AQL_task_basic():
    ex = ThreadPoolExecutor()
    t1 = AQLTask(lambda: 2, executor=ex)
    t2 = AQLTask(lambda prev: prev+2, dependencies=[t1], executor=ex, pipe_dependencies=True)
    t2._start()
    t1._start()
    assert t2.result(timeout=1) == 4


def test_AQL_task_result_passing():
    """
    Test if values from dependencies are passed in the correct order, i.e if dependencies=[dep1,dep2], [res(dep1),res(dep2)] is passed to the task function.
    """
    ex = ThreadPoolExecutor()
    t_string = AQLTask(lambda: "Value:", executor=ex)
    t_int = AQLTask(lambda: 42, executor=ex)
    t_concat = AQLTask(lambda s, i: s + str(i), dependencies=[t_string, t_int], executor=ex, pipe_dependencies=True)

    for t in [t_string, t_concat, t_int]:
        t._start()

    assert t_concat.result(timeout=1) == "Value:42"
