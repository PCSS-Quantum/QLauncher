import time
import functools
import pytest

import psutil

from quantum_launcher.base.base import Result
from quantum_launcher.launcher.aql import AQL, AQLTask
from quantum_launcher.problems import EC, TSP
from quantum_launcher.routines.dwave_routines import DwaveSolver, SimulatedAnnealingBackend
from quantum_launcher.routines.qiskit_routines import QAOA, QiskitBackend
from quantum_launcher.hampy import Equation

from dimod import SampleSet


def check_subprocesses_exit(max_timeout=5):
    def wrapper1(func):
        """Verify if test kills all its children :)"""
        @functools.wraps(func)
        def wrapper2(*args, **kwargs):
            current_process = psutil.Process()
            def curr_nc(): return len(current_process.children(recursive=True))
            num_children = curr_nc()

            func(*args, **kwargs)

            # Hacky, but killing a process from the os side might take some time.
            i = 0
            while i < max_timeout:
                i += 0.1
                time.sleep(0.1)
                if curr_nc() == num_children:
                    return
            assert curr_nc() == num_children
        return wrapper2
    return wrapper1


def prepare_AQL(mode='default') -> AQL:
    aql = AQL(mode)

    be = QiskitBackend('local_simulator')
    be.is_device = True
    t1 = aql.add_task((TSP.generate_tsp_instance(5), QAOA(), be))
    t2 = aql.add_task((TSP.generate_tsp_instance(5), QAOA(), be), dependencies=[t1])

    return aql


@check_subprocesses_exit()
def test_AQL_cancels_tasks():
    aql = prepare_AQL()

    aql.start()
    time.sleep(0.5)
    aql.cancel_running_tasks()

    for t in aql.tasks:
        assert t.cancelled()

    assert aql.results() == [None] * len(aql.tasks)

    with pytest.raises(ValueError):
        aql.start()


@check_subprocesses_exit()
def test_AQL_cancels_tasks_in_opt_mode():
    aql = prepare_AQL('optimize_session')
    aql.start()
    time.sleep(0.5)
    aql.cancel_running_tasks()

    for t in aql.tasks:
        assert t.cancelled()

    assert aql.results() == [None] * len(aql.tasks)

    with pytest.raises(ValueError):
        aql.start()


@check_subprocesses_exit()
def test_AQL_cancels_tasks_after_timeout():
    aql = prepare_AQL()

    aql.start()
    time.sleep(0.1)
    with pytest.raises(TimeoutError):
        aql.results(0.1, cancel_tasks_on_timeout=True)

    for t in aql._classical_tasks + aql._quantum_tasks:
        assert t.cancelled()


@check_subprocesses_exit()
def test_AQL_individual_tasks():
    aql = AQL()

    aql.add_task((EC.from_preset('micro'), QAOA(), QiskitBackend('local_simulator')))
    aql.add_task((EC.from_preset('micro'), DwaveSolver(), SimulatedAnnealingBackend('local_simulator')))

    aql.start()

    res = aql.results()
    assert len(res) == 2
    for r in res:
        assert isinstance(r, Result)
    assert isinstance(res[0].result, dict)
    assert isinstance(res[1].result, SampleSet)


@check_subprocesses_exit()
def test_AQL_context_manager():
    tasks: list[AQLTask] = []
    with AQL() as aql:
        t1 = aql.add_task((EC.from_preset('toy'), QAOA(), QiskitBackend('local_simulator')))
        t2 = aql.add_task((EC.from_preset('toy'), DwaveSolver(), SimulatedAnnealingBackend('local_simulator')))
        tasks: list[AQLTask] = [t1, t2]
        aql.start()

    for t in tasks:
        assert not t.running()
        assert t.result() is None


@check_subprocesses_exit()
def test_AQL_binds_params():
    aql = AQL('optimize_session')
    be = QiskitBackend('local_simulator')
    be.is_device = True
    aql.add_task((TSP.generate_tsp_instance(3), QAOA(), be), onehot='quadratic')

    aql.start()
    aql.results(10)
    t_gen = aql._classical_tasks[0]

    eq = Equation(t_gen.result())

    assert eq.is_quadratic()


@check_subprocesses_exit()
def test_AQL_session_optimization():
    classical_backend = QiskitBackend('local_simulator')
    totally_real_backend = QiskitBackend('local_simulator')
    totally_real_backend.is_device = True

    aql = AQL(mode='optimize_session')

    t1_temp = (EC.from_preset('micro'), QAOA(), totally_real_backend)
    t2_temp = (EC.from_preset('micro'), QAOA(), totally_real_backend)
    t3_temp = (EC.from_preset('micro'), QAOA(), classical_backend)

    order = []
    t1 = aql.add_task(t3_temp)
    t2 = aql.add_task(t1_temp, dependencies=[t1])
    t4 = aql.add_task(t3_temp, dependencies=[t2])
    t3 = aql.add_task(t2_temp, dependencies=[t2])

    for t in aql.tasks:
        t._inner_task.callbacks.append(order.append)

    assert aql._quantum_tasks == [t2, t3]
    assert len(aql._classical_tasks) == 4
    assert aql.tasks == [t1, t2, t4, t3]

    aql.start()
    aql.results()
    assert order == [t1.result(), t2.result(), t3.result(), t4.result()]
    del order


@check_subprocesses_exit()
def test_AQL_task_basic():
    t1 = AQLTask(lambda: 2)
    t2 = AQLTask(lambda prev: prev+2, dependencies=[t1], pipe_dependencies=True)
    t2.start()
    t1.start()
    assert t2.result(timeout=1) == 4


@check_subprocesses_exit()
def test_AQL_task_result_passing():
    """
    Test if values from dependencies are passed in the correct order, 
    i.e if dependencies=[dep1,dep2], [res(dep1),res(dep2)] is passed to the task function.
    """
    t_string = AQLTask(lambda: "Value:")
    t_int = AQLTask(lambda: 42)
    t_concat = AQLTask(lambda s, i: s + str(i), dependencies=[t_string, t_int], pipe_dependencies=True)

    for t in [t_string, t_concat, t_int]:
        t.start()

    assert t_concat.result(timeout=1) == "Value:42"


@check_subprocesses_exit()
def test_AQL_task_raises_error_from_target_fn():
    def err():
        raise ValueError

    t_err = AQLTask(err)

    with pytest.raises(ValueError):
        t_err.start()
        t_err.result()


@check_subprocesses_exit()
def test_task_dies_after_timeout_error():
    t = AQLTask(lambda: time.sleep(20))
    t.start()

    with pytest.raises(TimeoutError):
        t.result(0.1)


@check_subprocesses_exit()
def test_task_dies_after_going_out_of_scope():
    t = AQLTask(lambda: time.sleep(20))
    t.start()
