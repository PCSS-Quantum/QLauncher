"""All problems together"""

from qlauncher.base import Problem

from .optimization import EC, JSSP, QATM, TSP, GraphColoring, Knapsack, MaxCut
from .other import DTQW_1D, TabularML

Raw, _Circuit = [None] * 2

__all__ = ['Problem', 'MaxCut', 'EC', 'QATM', 'JSSP', 'TSP', 'GraphColoring', 'DTQW_1D', 'TabularML', 'Knapsack', '_Circuit']
