""" _summary_
"""
from typing import Literal
from cirq import Circuit, Sampler
from cirq import StabilizerSampler
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import BaseSamplerV2
from quantum_launcher.base import Backend


class CirqSampler(BaseSamplerV2):
    """_summary_

    Args:
        BaseSamplerV2 (_type_): _description_
    """

    def __init__(self, cirq_sampler: Sampler):
        pass


class CirqBackend(Backend):
    """

    Args:
        Backend (_type_): _description_
    """

    def __init__(self, name: Literal['local'] = 'local'):
        self.sampler = StabilizerSampler()
        self.optimizer = COBYLA()
