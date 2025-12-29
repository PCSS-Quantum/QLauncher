from qlauncher.base.base import Algorithm, Result
from qlauncher.problems.circuit import _Circuit
from qlauncher.routines.cirq import CirqBackend
from qlauncher.routines.qiskit.backends.qiskit_backend import QiskitBackend


class CircuitRunner(Algorithm[_Circuit, QiskitBackend | CirqBackend]):
	def __init__(self, shots: int) -> None:
		self.shots = shots

	def run(self, problem: _Circuit, backend: QiskitBackend | CirqBackend) -> Result:
		counts = backend.sample_circuit(problem.circuit)

		energy_fake_distribution = dict.fromkeys(counts.keys(), 1.0)  # No energy calculations in sampling

		return Result.from_counts_energies(counts, energy_fake_distribution)
