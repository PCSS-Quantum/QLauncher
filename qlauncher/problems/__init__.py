"""All problems together"""

from qlauncher.base import Problem

from .optimization import EC, JSSP, QATM, TSP, GraphColoring, Knapsack, MaxCut
from .other import TabularML

_Circuit = None

__all__ = ['Problem', 'MaxCut', 'EC', 'QATM', 'JSSP', 'TSP', 'GraphColoring', 'TabularML', 'Knapsack', '_Circuit']
