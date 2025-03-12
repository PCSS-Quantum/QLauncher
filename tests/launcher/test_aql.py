from concurrent.futures import ThreadPoolExecutor

import pytest

import numpy as np

from quantum_launcher.base.base import Result
from quantum_launcher.launcher.aql import AQL, AQLTask
from quantum_launcher.launcher.qlauncher import QuantumLauncher
from quantum_launcher.problems import MaxCut, EC
from quantum_launcher.routines.dwave_routines import DwaveSolver, SimulatedAnnealingBackend
from quantum_launcher.routines.qiskit_routines import QAOA, QiskitBackend
from quantum_launcher.routines.orca_routines import BBS, OrcaBackend


def test_AQL_individual_tasks():
    aql = AQL()

    aql.add_task((MaxCut.from_preset('default'), QAOA(), QiskitBackend('local_simulator')))
    aql.add_task((MaxCut.from_preset('default'), BBS(), OrcaBackend('local_simulator')))
    aql.add_task((EC.from_preset('micro'), DwaveSolver(), SimulatedAnnealingBackend('local_simulator')))

    aql.start()

    res = aql.get_results()
    assert len(res) == 3
    for r in res:
        assert isinstance(r, Result)


def test_AQL_chained_tasks():
    return
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

    assert len(aql.get_results()[0]) == 5
    assert order == wanted


def test_AQL_session_optimization():
    classical_backend = QiskitBackend('local_simulator')
    totally_real_backend = QiskitBackend('local_simulator')
    totally_real_backend.is_device = True

    aql = AQL(mode='optimize_session')

    t1_temp = (MaxCut.from_preset('default'), QAOA(), totally_real_backend)
    t2_temp = (MaxCut.from_preset('default'), QAOA(), totally_real_backend)
    t3_temp = (EC.from_preset('micro'), QAOA(), classical_backend)

    order = []
    t3 = aql.add_task(t3_temp)
    t1 = aql.add_task(t1_temp, dependencies=[t3])
    t2 = aql.add_task(t2_temp, dependencies=[t1])
    t4 = aql.add_task(t3_temp, dependencies=[t1])

    t3.__repr__ = lambda: 't3'
    t1.__repr__ = lambda: 't1'
    t2.__repr__ = lambda: 't2'
    t4.__repr__ = lambda: 't4'

    for t in aql.tasks:
        t.callbacks.append(order.append)

    assert aql.quantum_tasks == [t1, t2]
    assert len(aql.classical_tasks) == 4
    assert aql.tasks == [t3, t1, t2, t4]

    aql.start()
    aql.wait_for_finish(20)
    assert order == [t3, t1, t2, t4]


def test_AQL_task_basic():
    ex = ThreadPoolExecutor()
    t1 = AQLTask(lambda: 2, executor=ex)
    t2 = AQLTask(lambda prev: prev+2, dependencies=[t1], executor=ex, pipe_dependencies=True)
    t2.start()
    t1.start()
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
        t.start()

    assert t_concat.result(timeout=1) == "Value:42"
