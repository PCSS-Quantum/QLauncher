"""All problems together"""

from qlauncher.base import Problem

from .optimization import EC, QATM, MaxCut
from .other import TabularML

JSSP, TSP, GraphColoring, Knapsack, Raw, _Circuit = [None] * 6

__all__ = ['Problem', 'Raw', 'MaxCut', 'EC', 'QATM', 'JSSP', 'TSP', 'GraphColoring', 'TabularML', 'Knapsack', '_Circuit']
