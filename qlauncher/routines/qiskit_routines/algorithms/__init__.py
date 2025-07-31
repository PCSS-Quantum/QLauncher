""" Algorithms for qiskit """
from .qiskit_native import QAOA, FALQON
from .qml import TrainQSVCKernel
try:
    from .educated_guess import EducatedGuess
except ImportError:
    EducatedGuess = None

__all__ = ['QAOA', 'FALQON', 'EducatedGuess', 'TrainQSVCKernel']
