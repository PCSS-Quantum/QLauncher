""" Algorithms for qiskit """
from .qiskit_native import QAOA,LidAngleQAOA, FALQON
try:
    from .qml import TrainQSVCKernel
except ImportError:
    TrainQSVCKernel = None
try:
    from .educated_guess import EducatedGuess
except ImportError:
    EducatedGuess = None

__all__ = ['QAOA','LidAngleQAOA', 'FALQON', 'EducatedGuess', 'TrainQSVCKernel']
