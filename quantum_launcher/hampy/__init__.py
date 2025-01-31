""" Hampy is small package for creating Hamiltonian from boolean expressions """
from .hamiltonian import *
from .debug import *
from .quadratic import *
from .object import HampyEquation, HampyVariable
from .equations import one_in_n

__all__ = ['HampyEquation', 'HampyVariable', 'one_in_n']
