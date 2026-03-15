"""Wrapper problem for simple circuit running"""

from qiskit.primitives.containers.sampler_pub import SamplerPubLike

from qlauncher.base import Model
from qlauncher.routines.circuits import CIRCUIT_FORMATS


class _Circuit(Model):
    def __init__(self, circuit: CIRCUIT_FORMATS) -> None:
        self.variant = 'Circuit'
        self.circuit = circuit
