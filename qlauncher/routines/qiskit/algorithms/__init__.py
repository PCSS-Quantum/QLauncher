"""Algorithms for qiskit"""

from .qiskit_native import FALQON, QAOA
from .wrapper import _CircuitRunner

try:
	from .qml import TrainQSVCKernel
except ImportError:
	TrainQSVCKernel = None
try:
	from .educated_guess import EducatedGuess
except ImportError:
	EducatedGuess = None
try:
	from .quantum_walk import DiscreteTimeQuantumWalk
except ImportError:
	DiscreteTimeQuantumWalk = None

__all__ = ['QAOA', 'FALQON', 'EducatedGuess', 'TrainQSVCKernel', 'DiscreteTimeQuantumWalk', '_CircuitRunner']
