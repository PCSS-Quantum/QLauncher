""" Hampy is small package for creating Hamiltonian from boolean expressions """
from .object import Equation, Variable
from .equations import one_in_n
from .debug import TruthTable

__all__ = ['Equation', 'Variable', 'one_in_n', 'TruthTable']
