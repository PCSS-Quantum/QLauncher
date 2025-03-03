from collections import defaultdict
from .base import Problem
from typing import Dict, Callable, Any
from inspect import signature
import networkx as nx
import warnings

__QL_ADAPTERS: Dict[str, Dict[str, Callable]] = defaultdict(lambda: {})  # adapters[to][from]
__QL_FORMATTERS: Dict[type[Problem] | None, Dict[str, Callable]] = defaultdict(lambda: {})  # formatters[problem][format]


def _get_callable_name(c: Callable) -> str:
    """
    Gets name of input callable, made it because callable classes don't have a __name__
    """
    try:
        return c.__name__
    except AttributeError:
        return str(c.__class__)


class ProblemFormatter:
    """
    Converts input problem to a given format (input and output types determined by formatter and adapters in __init__)

    Probably shouldn't be constructed directly, call :py:func:`get_formatter()`
    """

    def __init__(self, formatter: Callable, adapters: list[Callable] | None = None):
        self.formatter = formatter
        self.adapters = adapters if adapters is not None else []

        self.formatter_sig = signature(self.formatter)

        self.run_params = {}

    def _formatter_call(self, run_params, *args, **kwargs):
        params_used = {k: v for k, v in run_params.items() if k in self.formatter_sig.parameters}

        unused_count = len(run_params) - len(params_used)
        if unused_count > 0:
            warnings.warn(
                f"{unused_count} unused parameters. {_get_callable_name(self.formatter)} does not accept {[k for k in run_params if not k in params_used]}", Warning)
        return self.formatter(*args, **kwargs, **params_used)

    def __call__(self, *args, **kwargs):
        # Reset bound params
        curr_run_params = dict(self.run_params)
        self.run_params = {}

        out = self._formatter_call(curr_run_params, *args, **kwargs)
        for adapter in self.adapters:
            out = adapter(out)

        return out

    def get_pipeline(self) -> str:
        """
        Returns:
            String representing the conversion process: problem -> formatter -> adapters (if applicable)
        """
        return " -> ".join(
            [str(list(self.formatter_sig.parameters.keys())[0])] +
            [_get_callable_name(self.formatter)] +
            [_get_callable_name(fn) for fn in self.adapters]
        )

    def set_run_param(self, param: str, value: Any) -> None:
        """
        Sets a parameter to be used during next conversion.

        Args:
            param: parameter key
            value: parameter value
        """
        self.run_params[param] = value

    def set_run_params(self, params: dict[str, Any]) -> None:
        """
        Sets multiple parameters to be used during next conversion.

        Args:
            params: parameters to be set
        """
        for k, v in params.items():
            self.set_run_param(k, v)


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


def register_formatter(problem: type[Problem] | None, alg_format: str):
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


def _find_shortest_adapter_path(problem: type[Problem], alg_format: str) -> list[str] | None:
    """
    Creates directed graph of possible conversions between formats and finds shortest path of formats between problem and alg_format.

    Returns:
        List of formats or None if no path was found.
    """
    G = nx.DiGraph()
    for problem_node in set(__QL_FORMATTERS[problem].keys()):
        G.add_edge("__problem__", problem_node)

    for out_form in set(__QL_ADAPTERS.keys()):
        for in_form in set(__QL_ADAPTERS[out_form].keys()):
            G.add_edge(in_form, out_form)

    path = nx.shortest_path(G, "__problem__", alg_format)
    assert isinstance(path, list) or path is None, "Something went wrong in `nx.shortest_path`"
    return path


def get_formatter(problem: type[Problem], alg_format: str) -> ProblemFormatter:
    """
    Creates a ProblemFormatter that converts a given Problem subclass into the requested format.

    Args:
        problem: Input problem type
        alg_format: Desired output format

    Returns:
        ProblemFormatter meeting the desired criteria.

    Raises:
        ValueError: If no combination of adapters can achieve conversion from problem to desired format.
    """
    formatter, adapters = None, None
    available_problem_formats = set(__QL_FORMATTERS[problem].keys())
    if alg_format in available_problem_formats:
        formatter = __QL_FORMATTERS[problem][alg_format]
    else:
        path = _find_shortest_adapter_path(problem, alg_format)

        if path is None:
            raise ValueError(f"No suitable ProblemFormatter can be found for combination of problem: {problem} and format: {alg_format}")

        formatter = __QL_FORMATTERS[problem][path[1]]
        adapters = []
        for i in range(1, len(path)-1):
            adapters.append(__QL_ADAPTERS[path[i+1]][path[i]])

    return ProblemFormatter(formatter, adapters)


@register_formatter(None, 'none')
def default_formatter(problem: Problem):
    return problem.instance
