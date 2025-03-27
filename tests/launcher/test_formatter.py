from quantum_launcher.base.adapter_structure import get_formatter, default_formatter
from quantum_launcher.problems.problem_initialization import TSP


def test_unreachable_returns_default():
    formatter = get_formatter(TSP, 'unknown')

    assert formatter.formatter == default_formatter
