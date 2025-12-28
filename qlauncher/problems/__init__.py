"""All problems together"""

from qlauncher.base import Problem

from .optimization import EC, JSSP, QATM, TSP, GraphColoring, Knapsack, MaxCut
from .other import DTQW_ND, TabularML

Raw, _Circuit = [None] * 2

__all__ = ['Problem', 'MaxCut', 'EC', 'QATM', 'JSSP', 'TSP', 'GraphColoring', 'DTQW_ND', 'TabularML', 'Knapsack', '_Circuit']
