"""All problems together"""

from qlauncher.base import Problem

from . import problem_formulations as _
from .ec import EC
from .graph_coloring import GraphColoring
from .knapsack import Knapsack
from .maxcut import MaxCut
from .problem_initialization import JSSP, Raw, TabularML, _Circuit
from .qatm import QATM
from .tsp import TSP

__all__ = ['Problem', 'Raw', 'MaxCut', 'EC', 'QATM', 'JSSP', 'TSP', 'GraphColoring', 'TabularML', 'Knapsack', '_Circuit']
