""" Hampy is a small package for creating Hamiltonians from boolean expressions """
from .object import Equation, Variable
from .equations import one_in_n
from .debug import TruthTable
from .utils import shift_affected_qubits

__all__ = ['Equation', 'Variable', 'one_in_n', 'TruthTable', 'shift_affected_qubits']
