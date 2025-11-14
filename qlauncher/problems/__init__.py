"""All problems together"""

from qlauncher.base import Problem

from . import problem_formulations as _
from .problem_initialization import EC, JSSP, QATM, TSP, GraphColoring, Knapsack, MaxCut, Molecule, Raw, TabularML, _Circuit

__all__ = ['Problem', 'Raw', 'MaxCut', 'EC', 'QATM', 'JSSP', 'TSP', 'GraphColoring', 'TabularML', 'Molecule', 'Knapsack', '_Circuit']
