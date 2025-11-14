from qlauncher.base.adapter_structure import default_formatter, get_formatter
from qlauncher.problems.problem_initialization import TSP


def test_unreachable_returns_default():
	formatter = get_formatter(TSP, 'unknown')

	assert formatter.formatter == default_formatter
