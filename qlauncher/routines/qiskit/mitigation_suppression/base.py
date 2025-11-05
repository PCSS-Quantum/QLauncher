from abc import ABC, abstractmethod
from qiskit import QuantumCircuit

from qlauncher.base import Result
from qlauncher.routines.qiskit import QiskitBackend
# from qlauncher.routines.cirq import CirqBackend


class CircuitModificationMethod(ABC):
    """Method that relies only on modifying the input circuit"""

    @abstractmethod
    def modify_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        pass


class CircuitSamplingMethod(ABC):
    """Method that relies on sampling multiple modified versions of the circuit"""

    @abstractmethod
    def sample(self, circuit: QuantumCircuit, backend: QiskitBackend) -> Result:
        pass
