from qlauncher.problems.problem_initialization import TabularML
from qlauncher.base import formatter


@formatter(TabularML, 'tabular_ml')
def tabular_formatter(problem: TabularML):
    return problem.instance
