from abc import ABC, abstractmethod

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

# from qlauncher.routines.cirq import CirqBackend


class CircuitModificationMethod(ABC):
	"""Method that relies only on modifying the input circuit"""

	@abstractmethod
	def modify_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
		pass


class CircuitExecutionMethod(ABC):
	"""Method that relies on executing multiple modified versions of the circuit"""

	@abstractmethod
	def sample(self, circuit: QuantumCircuit, backend: 'QiskitBackend', shots: int = 1024) -> dict[str, int]:
		"""
		Sample circuit on the backend.

		Args:
		    circuit (QuantumCircuit): Circuit to run.
		    backend (QiskitBackend): Backend to run on.
		    shots (int, optional): Number of samples to collect. Defaults to 1024.

		Returns:
		    dict[str,int]: Bitstring counts.
		"""
		pass

	@abstractmethod
	def estimate(self, circuit: QuantumCircuit, observable: SparsePauliOp, backend: 'QiskitBackend') -> float:
		"""
		Estimate energy of observable after running a given circuit on the backend.

		Args:
		    circuit (QuantumCircuit): Circuit to run.
		    observable (SparsePauliOp): Observable to estimate.
		    backend (QiskitBackend): Backend to use.

		Returns:
		    float: Estimated energy of the observable.
		"""
		pass
