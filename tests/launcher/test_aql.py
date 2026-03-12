import time
from typing import Literal

import pytest
from dimod import SampleSet

from qlauncher.base.base import Result
from qlauncher.launcher.aql import AQL, ManagerBackedTask
from qlauncher.routines.dwave import SimulatedAnnealingBackend
from qlauncher.routines.dwave.algorithms import SimulatedAnnealing
from qlauncher.routines.qiskit import QAOA, QiskitBackend
from qlauncher.workflow.local_scheduler import LocalJobManager
from tests.utils.multiprocessing import check_subprocesses_exit
from tests.utils.problem import get_hamiltonian, get_qubo


def prepare_AQL(mode: Literal['default', 'optimize_session'] = 'default', long: bool = False) -> AQL:
	qaoa_p = 10 if long else 1
	aql = AQL(mode)

	be = QiskitBackend('local_simulator')
	be.is_device = True
	t1 = aql.add_task((get_hamiltonian(), QAOA(p=qaoa_p), be))
	aql.add_task((get_hamiltonian(), QAOA(p=qaoa_p), be), dependencies=[t1])

	return aql


@check_subprocesses_exit()
def test_AQL_cancels_tasks() -> None:
	aql = prepare_AQL()

	aql.start()
	time.sleep(0.5)
	aql.cancel_running_tasks()

	for t in aql.tasks:
		assert t.cancelled()

	with pytest.raises(ValueError):
		aql.start()


@check_subprocesses_exit()
def test_AQL_cancels_tasks_in_opt_mode() -> None:
	aql = prepare_AQL('optimize_session', long=True)
	aql.start()
	time.sleep(0.5)
	aql.cancel_running_tasks()

	for t in aql.tasks:
		assert t.cancelled()

	assert aql.results() == [None] * len(aql.tasks)

	with pytest.raises(ValueError):
		aql.start()


@check_subprocesses_exit()
def test_AQL_cancels_tasks_after_timeout() -> None:
	aql = prepare_AQL()

	aql.start()

	with pytest.raises(TimeoutError):
		aql.results(0.01, cancel_tasks_on_timeout=True)

	for t in aql._classical_tasks + aql._quantum_tasks:
		assert t.cancelled()


@check_subprocesses_exit()
def test_AQL_individual_tasks() -> None:
	aql = AQL()

	aql.add_task((get_hamiltonian(), QAOA(), QiskitBackend('local_simulator')))
	aql.add_task((get_qubo(), SimulatedAnnealing(num_reads=10), SimulatedAnnealingBackend()))

	aql.start()

	res = aql.results()
	assert len(res) == 2
	for r in res:
		assert isinstance(r, Result)
	assert isinstance(res[0].result, dict)
	assert isinstance(res[1].result, SampleSet)


@check_subprocesses_exit()
def test_AQL_context_manager() -> None:
	tasks: list[ManagerBackedTask] = []
	with AQL() as aql:
		t1 = aql.add_task((get_hamiltonian(), QAOA(p=10), QiskitBackend('local_simulator')))
		t2 = aql.add_task((get_qubo(), SimulatedAnnealing(num_reads=10000), SimulatedAnnealingBackend()))
		tasks: list[ManagerBackedTask] = [t1, t2]
		aql.start()

	for t in tasks:
		assert not t.running()
		assert t.result() is None


@check_subprocesses_exit()
def test_AQL_session_optimization() -> None:
	classical_backend = QiskitBackend('local_simulator')
	totally_real_backend = QiskitBackend('local_simulator')
	totally_real_backend.is_device = True

	aql = AQL(mode='optimize_session')

	t1_temp = (get_hamiltonian(), QAOA(), totally_real_backend)
	t2_temp = (get_hamiltonian(), QAOA(), totally_real_backend)
	t3_temp = (get_hamiltonian(), QAOA(), classical_backend)

	order = []
	t1 = aql.add_task(t3_temp)
	t2 = aql.add_task(t1_temp, dependencies=[t1])
	t4 = aql.add_task(t3_temp, dependencies=[t2])
	t3 = aql.add_task(t2_temp, dependencies=[t2])

	for t in aql.tasks:
		t.callbacks.append(order.append)

	assert aql._quantum_tasks == [t2, t3]
	assert len(aql._classical_tasks) == 4
	assert aql.tasks == [t1, t2, t4, t3]

	aql.start()
	aql.results()
	assert order == [t1.result(), t2.result(), t3.result(), t4.result()]
	del order


@check_subprocesses_exit()
def test_AQL_task_basic() -> None:
	manager = LocalJobManager()
	t1 = ManagerBackedTask(lambda: 2, manager=manager)
	t2 = ManagerBackedTask(lambda prev: prev + 2, dependencies=[t1], pipe_dependencies=True, manager=manager)
	t2.start()
	t1.start()
	assert t2.result(timeout=1) == 4


@check_subprocesses_exit()
def test_AQL_task_result_passing() -> None:
	"""
	Test if values from dependencies are passed in the correct order,
	i.e if dependencies=[dep1,dep2], [res(dep1),res(dep2)] is passed to the task function.
	"""
	manager = LocalJobManager()
	t_string = ManagerBackedTask(lambda: 'Value:', manager=manager)
	t_int = ManagerBackedTask(lambda: 42, manager=manager)
	t_concat = ManagerBackedTask(lambda s, i: s + str(i), dependencies=[t_string, t_int], pipe_dependencies=True, manager=manager)

	for t in [t_string, t_concat, t_int]:
		t.start()

	assert t_concat.result(timeout=1) == 'Value:42'


@check_subprocesses_exit()
def test_AQL_task_raises_error_from_target_fn() -> None:
	manager = LocalJobManager()

	def err():
		raise ValueError

	t_err = ManagerBackedTask(err, manager=manager)

	with pytest.raises(ValueError):
		t_err.start()
		t_err.result()


@check_subprocesses_exit()
def test_task_dies_after_timeout_error() -> None:
	manager = LocalJobManager()
	t = ManagerBackedTask(lambda: time.sleep(20), manager=manager)
	t.start()

	with pytest.raises(TimeoutError):
		t.result(0.1)


@check_subprocesses_exit()
def test_task_dies_after_going_out_of_scope() -> None:
	manager = LocalJobManager()
	t = ManagerBackedTask(lambda: time.sleep(20), manager=manager)
	t.start()
