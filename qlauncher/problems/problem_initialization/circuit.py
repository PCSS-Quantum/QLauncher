"""Wrapper problem for simple circuit running"""

from qiskit.primitives.containers.sampler_pub import SamplerPubLike

from qlauncher.base import ProblemLike


class _Circuit(ProblemLike):
	def __init__(self, pub: SamplerPubLike) -> None:
		self.variant = 'Circuit'
		self.pub = pub
