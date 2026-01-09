from qlauncher.base import Algorithm, Result
from qlauncher.base.problem_like import GroverCircuit
from qlauncher.routines.qiskit.backends.qiskit_backend import QiskitBackend

from qiskit.circuit.library import grover_operator
from qiskit import QuantumCircuit


class Grover(Algorithm[GroverCircuit, QiskitBackend]):
	def __init__(self, n_shots: int, **alg_kwargs):
		super().__init__(**alg_kwargs)
		self.n_shots = n_shots

	def run(self, problem: GroverCircuit, backend: QiskitBackend) -> Result:
		grover_op = grover_operator(problem.oracle, problem.state_prep)
		full_circuit = QuantumCircuit(problem.oracle.num_qubits)

		full_circuit.append(problem.state_prep, range(problem.state_prep.num_qubits))
  
		for _ in range(problem.num_iterations):
			full_circuit.compose(grover_op, inplace=True)
		full_circuit.measure_all()
  
		self._circuit = full_circuit
  
		job = backend.sampler.run([full_circuit], shots=self.n_shots)
  
		return job.result()[0]

	def circuit(self, mode: str = '') -> QuantumCircuit | None:
		if not self._circuit:
			print("Run the algorithm to generate circuit")
		else:
			return self._circuit