from typing import Any
from abc import ABC, abstractmethod
import cirq
from cirq.contrib.qasm_import import circuit_from_qasm
from qiskit import QuantumCircuit, qasm2


class Translation(ABC):
    """ Translation layer for circuits written in different languages """
    translation_map = {}

    def __init__(self):
        pass

    def __init_subclass__(cls):
        Translation.translation_map[cls.__name__] = cls

    @abstractmethod
    def to_qasm(self, circuit: Any) -> str:
        pass

    @abstractmethod
    def from_qasm(self, qasm: str) -> Any:
        pass


class CirqTranslation(Translation):
    language: str = 'cirq'

    def to_qasm(self, circuit: cirq.Circuit) -> str:
        return circuit.to_qasm()

    def from_qasm(self, qasm: str) -> cirq.Circuit:
        return circuit_from_qasm(qasm)


class QiskitTranslation(Translation):
    language: str = 'qiskit'

    def to_qasm(self, circuit: QuantumCircuit) -> str:
        return qasm2.dumps(circuit)

    def from_qasm(self, qasm: str) -> QuantumCircuit:
        return QuantumCircuit.from_qasm_str(qasm)
