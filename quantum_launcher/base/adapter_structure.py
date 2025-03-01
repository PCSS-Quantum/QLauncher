from collections import defaultdict
from .base import Problem
from typing import Dict, Tuple, Callable


__QL_ADAPTERS: Dict[str, Dict[str, Callable]] = defaultdict(lambda: {})
__QL_FORMATTERS: Dict[type, Dict[str, Callable]] = defaultdict(lambda: {})


class FormatterParams(dict):
    DEFAULT_PARAMS = {
        'onehot': 'exact',
        'constraint_weight': 1,
        'optimization_weight': 1
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_defaults()

    def _set_defaults(self):
        for key, value in self.DEFAULT_PARAMS.items():
            self[key] = value

    def __getitem__(self, key):
        if key not in self.keys():
            raise ValueError(f"Parameter {key} not found in formatter")
        return super().__getitem__(key)


class ProblemFormatter:
    def __init__(self, formatter: Callable, adapter: Callable | None):
        self.formatter = formatter
        self.adapter = adapter
        self.run_params = FormatterParams()

    def __call__(self, *args, **kwargs):
        curr_run_params = dict(self.run_params)
        self.run_params._set_defaults()

        if self.adapter is not None:
            return self.adapter(self.formatter(*args, params=curr_run_params, **kwargs))

        return self.formatter(*args, params=curr_run_params, **kwargs)

    def set_run_param(self, param, value):
        self.run_params[param] = value


def register_adapter(translates_from: str, translates_to: str) -> Callable:
    """
    Register a function as an adapter from one problem format to another.

    Args:
        translates_from: Input format
        translates_to: Output format

    Returns:
        Same function
    """
    def decorator(func):
        if isinstance(func, type):
            func = func()
        __QL_ADAPTERS[translates_to][translates_from] = func
        return func
    return decorator


def register_formatter(problem: Problem, alg_format: str):
    """
    Register a function as a formatter for a given problem type to a given format.

    Args: 
        problem: Input problem type
        alg_format: Output format

    Returns:
        Same function
    """
    def decorator(func):
        if isinstance(func, type):
            func = func()
        __QL_FORMATTERS[problem][alg_format] = func
        return func
    return decorator


def get_formatter(problem: Problem, alg_format: str):
    formatter, adapter = None, None
    available_problem_formats = set(__QL_FORMATTERS[problem].keys())
    if alg_format in available_problem_formats:
        formatter = __QL_FORMATTERS[problem][alg_format]
    else:
        # Look for a viable pair of formats to translate between
        available_adapter_formats = set(__QL_ADAPTERS[alg_format].keys())
        matches = available_adapter_formats.intersection(available_problem_formats)

        if len(matches) == 0:
            raise ValueError(f"No suitable Problem Formatter can be found for combination of problem: {problem} and format: {alg_format}")

        raw_format = list(matches)[0]

        formatter, adapter = __QL_FORMATTERS[problem][raw_format], __QL_ADAPTERS[alg_format][raw_format]

    return ProblemFormatter(formatter, adapter)


@register_formatter(None, 'none')
def default_formatter(problem: Problem, params: FormatterParams = None):
    return problem.instance
