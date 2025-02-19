from typing import Literal
from cirq import Circuit, Sampler
from cirq import StabilizerSampler
from quantum_launcher.base import Algorithm, Backend
from qiskit_algorithms.optimizers import COBYLA


class CirqBackend(Backend):
    def __init__(self, name: Literal['local'] = 'local'):
        self.sampler = StabilizerSampler()
        self.optimizer = COBYLA()
