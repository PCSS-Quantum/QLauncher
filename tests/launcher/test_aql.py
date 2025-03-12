from concurrent.futures import ThreadPoolExecutor
import time

from quantum_launcher.base.base import Result
from quantum_launcher.launcher.aql import AQL, AQLTask
from quantum_launcher.problems import EC, TSP
from quantum_launcher.routines.dwave_routines import DwaveSolver, SimulatedAnnealingBackend
from quantum_launcher.routines.qiskit_routines import QAOA, QiskitBackend

from quantum_launcher.hampy import Equation


def test_AQL_individual_tasks():
    aql = AQL()

    aql.add_task((EC.from_preset('micro'), QAOA(), QiskitBackend('local_simulator')))
    aql.add_task((EC.from_preset('micro'), DwaveSolver(), SimulatedAnnealingBackend('local_simulator')))

    aql.start()

    res = aql.results()
    assert len(res) == 2
    for r in res:
        assert isinstance(r, Result)


def test_AQL_binds_params():
    aql = AQL('optimize_session')
    be = QiskitBackend('local_simulator')
    be.is_device = True
    aql.add_task((TSP.generate_tsp_instance(3), QAOA(), be), onehot='quadratic')

    aql.start()
    aql.wait_for_finish(10)
    t_gen = aql._classical_tasks[0]

    eq = Equation(t_gen.result())

    assert eq.is_quadratic()


def prepare_AQL(mode='default'):
    aql = AQL(mode)

    be = QiskitBackend('local_simulator')
    be.is_device = True
    t1 = aql.add_task((TSP.generate_tsp_instance(2), QAOA(), be))
    t2 = aql.add_task((TSP.generate_tsp_instance(2), QAOA(), be), dependencies=[t1])

    return aql


def test_AQL_cancels_tasks():
    aql = prepare_AQL()

    aql.start()
    time.sleep(0.1)
    aql.cancel_running_tasks()

    for t in aql.tasks:
        assert t.cancelled()

    assert aql.results() == [None] * len(aql.tasks)

    # Check if we can launch aql again
    aql.start()
    for r in aql.results(5):
        assert isinstance(r, Result)


def test_AQL_cancels_tasks_in_opt_mode():
    aql = prepare_AQL('optimize_session')
    aql.start()
    time.sleep(0.1)
    aql.cancel_running_tasks()

    for t in aql.tasks:
        assert t.cancelled()

    assert aql.results() == [None] * len(aql.tasks)

    # Check if we can launch aql again
    aql.start()
    for r in aql.results(5):
        assert isinstance(r, Result)


def test_AQL_session_optimization():
    classical_backend = QiskitBackend('local_simulator')
    totally_real_backend = QiskitBackend('local_simulator')
    totally_real_backend.is_device = True

    aql = AQL(mode='optimize_session')

    t1_temp = (EC.from_preset('micro'), QAOA(), totally_real_backend)
    t2_temp = (EC.from_preset('micro'), QAOA(), totally_real_backend)
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

    assert aql._quantum_tasks == [t1, t2]
    assert len(aql._classical_tasks) == 4
    assert aql.tasks == [t3, t1, t2, t4]

    aql.start()
    aql.wait_for_finish(10)
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
    Test if values from dependencies are passed in the correct order, 
    i.e if dependencies=[dep1,dep2], [res(dep1),res(dep2)] is passed to the task function.
    """
    ex = ThreadPoolExecutor()
    t_string = AQLTask(lambda: "Value:", executor=ex)
    t_int = AQLTask(lambda: 42, executor=ex)
    t_concat = AQLTask(lambda s, i: s + str(i), dependencies=[t_string, t_int], executor=ex, pipe_dependencies=True)

    for t in [t_string, t_concat, t_int]:
        t.start()

    assert t_concat.result(timeout=1) == "Value:42"
