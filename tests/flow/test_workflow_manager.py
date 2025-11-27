from qlauncher import QLauncher
from qlauncher.base import Algorithm
from qlauncher.base.problem_like import QUBO, ProblemLike
from qlauncher.problems import MaxCut
from qlauncher.workflow import WorkflowManager


def task1() -> int:
	return 10


def task2(value: int) -> int:
	return value * 3


def task3(result1: ProblemLike) -> float:
	return result1.instance * 3


def check_if_qubo(qubo: QUBO) -> bool:
	return isinstance(qubo, QUBO)


def test_task_addition() -> None:
	with WorkflowManager() as wm:
		data = wm.task(task1)
		result = wm.task(task2, (data,))
	wm()
	assert result.result == 30


def test_running() -> None:
	with WorkflowManager() as wm:
		data = wm.input(ProblemLike)
		result = wm.task(task3, (data,))
		wm.output(result)

	assert wm(ProblemLike(4)) == 12


def test_workflow() -> None:
	with WorkflowManager() as wm:
		data = wm.input(ProblemLike)
		result = wm.task(task3, (data,))
		wm.output(result)

	workflow = wm.to_workflow()
	assert isinstance(workflow, Algorithm)
	launcher = QLauncher(ProblemLike(20), workflow)
	result = launcher.run()
	assert result == 60


def test_workflow_format() -> None:
	with WorkflowManager() as wm:
		data = wm.input(format=QUBO)
		result = wm.task(check_if_qubo, (data,))
		wm.output(result)

	workflow = wm.to_workflow()
	assert workflow.get_input_format() is QUBO
	launcher = QLauncher(MaxCut.from_preset(instance_name='default'), workflow)
	assert launcher.run()
