""" Hampy is small package for creating Hamiltonian from boolean expressions """
from .hamiltonian import *
from .debug import *
from .quadratic import *
from .object import LogicalEquation, Variable
from .equations import one_in_n

__all__ = ['LogicalEquation', 'Variable', 'one_in_n']
