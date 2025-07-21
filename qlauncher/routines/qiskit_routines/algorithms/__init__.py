""" Algorithms for qiskit """
from .qiskit_native import QAOA, FALQON
try:
    from .educated_guess import EducatedGuess
except ImportError:
    EducatedGuess = None

__all__ = ['QAOA', 'FALQON', 'EducatedGuess']
