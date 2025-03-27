from collections import defaultdict
from typing import Dict, Callable, Any
from inspect import signature
import warnings
import networkx as nx
from .base import Problem

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


def _merge_dicts(dicts: list[dict]) -> dict:
    out = {}
    for d in dicts:
        out = out | d
    return out


class ProblemFormatter:
    """
    Converts input problem to a given format (input and output types determined by formatter and adapters in __init__)

    Probably shouldn't be constructed directly, call :py:func:`get_formatter()`
    """

    def __init__(self, formatter: Callable, adapters: list[Callable] | None = None):
        self.formatter = formatter
        self.adapters = [a['func'] for a in adapters] if adapters is not None else []
        self.adapter_requirements = _merge_dicts([a['formatter_requirements'] for a in adapters] if adapters is not None else [])

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

        common_params = set(curr_run_params.keys()).intersection(set(self.adapter_requirements.keys()))
        if len(common_params) > 0:
            warnings.warn(
                f"Attempting to reassign parameter values required by one of the adapters: {common_params}, those params will not be set.")

        curr_run_params = curr_run_params | self.adapter_requirements

        out = self._formatter_call(curr_run_params, *args, **kwargs)
        for a in self.adapters:
            out = a(out)

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
            param (str): parameter key
            value (str): parameter value
        """
        self.run_params[param] = value

    def set_run_params(self, params: dict[str, Any]) -> None:
        """
        Sets multiple parameters to be used during next conversion.

        Args:
            params (dict[str, Any]): parameters to be set
        """
        for k, v in params.items():
            self.set_run_param(k, v)


def adapter(translates_from: str, translates_to: str, **kwargs) -> Callable:
    """
    Register a function as an adapter from one problem format to another.

    Args:
        translates_from (str): Input format
        translates_to (str): Output format

    Returns:
        Same function
    """
    def decorator(func):
        if isinstance(func, type):
            func = func()
        __QL_ADAPTERS[translates_to][translates_from] = {'func': func, 'formatter_requirements': kwargs}
        return func
    return decorator


def formatter(problem: type[Problem] | None, alg_format: str):
    """
    Register a function as a formatter for a given problem type to a given format.

    Args:
        problem (type[Problem]): Input problem type
        alg_format (str): Output format

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
    for problem_node in __QL_FORMATTERS[problem]:
        G.add_edge("__problem__", problem_node)

    for out_form in __QL_ADAPTERS:
        for in_form in __QL_ADAPTERS[out_form]:
            G.add_edge(in_form, out_form)

    if not G.has_node(alg_format):
        return None

    path = nx.shortest_path(G, "__problem__", alg_format)
    assert isinstance(path, list) or path is None, "Something went wrong in `nx.shortest_path`"
    return path


def get_formatter(problem: type[Problem], alg_format: str) -> ProblemFormatter:
    """
    Creates a ProblemFormatter that converts a given Problem subclass into the requested format.

    Args:
        problem (type[Problem]): Input problem type
        alg_format (str): Desired output format

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
            formatter = default_formatter
        else:
            formatter = __QL_FORMATTERS[problem][path[1]]
            adapters = []
            for i in range(1, len(path)-1):
                adapters.append(__QL_ADAPTERS[path[i+1]][path[i]])

    return ProblemFormatter(formatter, adapters)


@formatter(None, 'none')
def default_formatter(problem: Problem):
    return problem.instance
