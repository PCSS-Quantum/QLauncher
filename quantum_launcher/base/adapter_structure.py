from collections import defaultdict
from .base import Problem
from typing import Dict, Tuple, Callable

__QL_TRANSLATIONS: Dict[type, Tuple[type, Callable]] = {}
__QL_ADAPTERS: Dict[type, Dict[type, Callable]] = defaultdict(lambda: {})

class FormatterParams(dict):
    DEFAULT_PARAMS = {
        'onehot': 'exact',
        'constraint_weight': 1,
        'optimization_weight': 1
    }
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        for key, value in self.DEFAULT_PARAMS.items():
            self[key] = value

    def __getitem__(self, key):
        if key not in self.keys():
            raise ValueError(f"Parameter {key} not found in formatter")
        return super().__getitem__(key)

class Formatter:
    def __init__(self, func):
        self.func = func
        self.run_params = FormatterParams()

    def __call__(self, *args, **kwargs):
        return self.func(*args,params=self.run_params, **kwargs)
    
    def set_param(self, param, value):
        self.run_params[param] = value

def adapter(translates_from: str, translates_to: str):
    def outer(func):
        def inner(func2):
            def inner_inner(*args, **kwargs):
                return func(func2(*args, **kwargs))
            return inner_inner
        __QL_TRANSLATIONS[translates_to] = (translates_from, inner)
        return inner
    return outer


def formatter(problem: Problem, output_format: str):
    def wrapper(func):
        if isinstance(func, type):
            func = func()
            
        formatter_obj = Formatter(func)
        __QL_ADAPTERS[problem][output_format] = formatter_obj 
        return formatter_obj 
    return wrapper


@formatter(None, 'none')
def default_formatter(problem: Problem):
    return problem.instance


def get_formatter(problem_id: Problem, alg_format: str):
    if alg_format in __QL_ADAPTERS[problem_id]:
        return __QL_ADAPTERS[problem_id][alg_format]
    if alg_format in __QL_TRANSLATIONS:
        origin_format, translation = __QL_TRANSLATIONS[alg_format]
        raw_formatter: Formatter = get_formatter(problem_id, origin_format)
        raw_formatter.func = translation(raw_formatter.func)
        return raw_formatter
    return default_formatter
