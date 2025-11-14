from qlauncher.base import formatter
from qlauncher.problems.problem_initialization import TabularML


@formatter(TabularML, 'tabular_ml')
def tabular_formatter(problem: TabularML):
	return problem.instance
