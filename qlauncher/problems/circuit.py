"""Wrapper problem for simple circuit running"""

from qiskit.primitives.containers.sampler_pub import SamplerPubLike

from qlauncher.base import ProblemLike
from qlauncher.routines.circuits import CIRCUIT_FORMATS


class _Circuit(ProblemLike):
	def __init__(self, circuit: CIRCUIT_FORMATS) -> None:
		self.variant = 'Circuit'
		self.circuit = circuit
