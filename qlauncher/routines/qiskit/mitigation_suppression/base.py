from abc import ABC, abstractmethod
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qlauncher.base import Result
# from qlauncher.routines.cirq import CirqBackend


class CircuitModificationMethod(ABC):
    """Method that relies only on modifying the input circuit"""

    @abstractmethod
    def modify_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        pass


class CircuitExecutionMethod(ABC):
    """Method that relies on executing multiple modified versions of the circuit"""

    @abstractmethod
    def sample(self, circuit: QuantumCircuit, backend: "QiskitBackend", shots: int = 1024) -> Result:
        pass

    @abstractmethod
    def estimate(self, circuit: QuantumCircuit, observable: SparsePauliOp, backend: "QiskitBackend") -> Result:
        pass
