from qlauncher.base.base import Algorithm, Result
from qlauncher.problems.circuit import _Circuit
from qlauncher.routines.qiskit.backends.gate_circuit_backend import GateCircuitBackend


class CircuitRunner(Algorithm[_Circuit, GateCircuitBackend]):
	def __init__(self, shots: int) -> None:
		self.shots = shots

	def run(self, problem: _Circuit, backend: GateCircuitBackend) -> Result:
		counts = backend.sample_circuit(problem.circuit)

		energy_fake_distribution = dict.fromkeys(counts.keys(), 1.0)  # No energy calculations in sampling

		return Result.from_counts_energies(counts, energy_fake_distribution)
