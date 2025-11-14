"""Algorithms for qiskit"""

from .qiskit_native import QAOA, FALQON
from .wrapper import _CircuitRunner

try:
	from .qml import TrainQSVCKernel
except ImportError:
	TrainQSVCKernel = None
try:
	from .educated_guess import EducatedGuess
except ImportError:
	EducatedGuess = None

__all__ = ['QAOA', 'FALQON', 'EducatedGuess', 'TrainQSVCKernel', '_CircuitRunner']
