"""All problems together"""

from qlauncher.base import Problem

from . import problem_formulations as _
from .ec import EC
from .maxcut import MaxCut
from .problem_initialization import JSSP, TSP, GraphColoring, Knapsack, Raw, TabularML, _Circuit
from .qatm import QATM

__all__ = ['Problem', 'Raw', 'MaxCut', 'EC', 'QATM', 'JSSP', 'TSP', 'GraphColoring', 'TabularML', 'Knapsack', '_Circuit']
