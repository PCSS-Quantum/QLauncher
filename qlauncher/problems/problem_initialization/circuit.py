"""Wrapper problem for simple circuit running"""

from qiskit.primitives.containers.sampler_pub import SamplerPubLike

from qlauncher.base.base import Problem


class _Circuit(Problem):
	def __init__(self, pub: SamplerPubLike, shots: int, instance_name: str = 'unnamed') -> None:
		super().__init__({'pub': pub, 'shots': shots}, instance_name)
		self.variant = 'Circuit'
