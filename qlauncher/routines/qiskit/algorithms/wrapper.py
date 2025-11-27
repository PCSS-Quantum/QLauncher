from collections.abc import Callable

from qiskit.primitives import BitArray
from qiskit.primitives.containers.sampler_pub import SamplerPub

from qlauncher.base.base import Algorithm, Backend, Problem, Result
from qlauncher.problems.circuit import _Circuit
from qlauncher.routines.cirq import CirqBackend
from qlauncher.routines.qiskit.backends.qiskit_backend import QiskitBackend
from qlauncher.utils import int_to_bitstring


class _CircuitRunner(Algorithm):
	_algorithm_format = 'none'

	def __init__(self, **alg_kwargs) -> None:
		super().__init__(**alg_kwargs)

	def run(self, problem: Problem, backend: Backend, formatter: Callable) -> Result:
		if not isinstance(problem, _Circuit):
			raise ValueError('_CircuitRunner can only process _Circuit problems.')

		if not isinstance(backend, QiskitBackend | CirqBackend):
			raise ValueError('A gate-based backend is needed to run circuits.')

		instance = formatter(problem)

		num_bits = SamplerPub.coerce(instance['pub']).circuit.num_qubits

		out = backend.sampler.run([instance['pub']], shots=instance['shots']).result()[0]
		data = BitArray.concatenate_bits(list(out.data.values()))

		counts = data.get_int_counts()
		counts = {int_to_bitstring(k, total_bits=num_bits): v for k, v in counts.items()}

		energy_fake_distribution = dict.fromkeys(counts.keys(), 1.0)  # No energy calculations in sampling

		return Result.from_counts_energies(counts, energy_fake_distribution)


class CircuitRunner(Algorithm[_Circuit, QiskitBackend]):
	def __init__(self, shots: int) -> None:
		self.shots = shots

	def run(self, problem: _Circuit, backend: QiskitBackend) -> Result:
		num_bits = SamplerPub.coerce(problem.pub).circuit.num_qubits

		out = backend.sampler.run([problem.pub], shots=self.shots).result()[0]
		data = BitArray.concatenate_bits(list(out.data.values()))

		counts = data.get_int_counts()
		counts = {int_to_bitstring(k, total_bits=num_bits): v for k, v in counts.items()}

		energy_fake_distribution = dict.fromkeys(counts.keys(), 1.0)  # No energy calculations in sampling

		return Result.from_counts_energies(counts, energy_fake_distribution)
