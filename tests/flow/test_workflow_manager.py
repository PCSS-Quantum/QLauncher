from qlauncher import QLauncher
from qlauncher.base import Algorithm
from qlauncher.problems import MaxCut
from qlauncher.workflow import WorkflowManager


def task1() -> int:
	return 10


def task2(result1):
	return result1 * 3


def check_if_qubo(qubo) -> int:
	import numpy as np

	assert isinstance(qubo[0], np.ndarray)
	return 1


def test_task_addition() -> None:
	with WorkflowManager() as wm:
		data = wm.task(task1)
		result = wm.task(task2, (data,))
	wm()
	assert result.result == 30


def test_running() -> None:
	with WorkflowManager() as wm:
		data = wm.input()
		result = wm.task(task2, (data,))
		wm.output(result)

	assert wm(4) == 12


def test_workflow() -> None:
	with WorkflowManager() as wm:
		data = wm.input(format='qubo')
		result = wm.task(task2, (data,))
		wm.output(result)

	workflow = wm.to_workflow()
	assert isinstance(workflow, Algorithm)
	launcher = QLauncher(20, workflow)
	result = launcher.run()
	assert result == 60


def test_workflow_format() -> None:
	with WorkflowManager() as wm:
		data = wm.input(format='qubo')
		result = wm.task(check_if_qubo, (data,))
		wm.output(result)

	workflow = wm.to_workflow()
	assert workflow._algorithm_format == 'qubo'
	launcher = QLauncher(MaxCut.from_preset(instance_name='default'), workflow)
	result = launcher.run()
	assert result == 1
