from typing import Any, Dict, Type, TypeVar
from abc import ABC, abstractmethod
import cirq
from cirq.contrib.qasm_import import circuit_from_qasm
import qiskit
from qiskit import qasm2

Circuit = TypeVar('Circuit')


class Translation(ABC):
    """ Translation layer for circuits written in different languages """
    language: str = 'None'
    circuit_object: Type = None
    translation_map: Dict[str, "Translation"] = {}
    object_map: Dict[Circuit, "Translation"] = {}

    def __init_subclass__(cls):
        Translation.translation_map[cls.language] = cls()
        Translation.object_map[cls.circuit_object] = cls()

    @abstractmethod
    def to_qasm(self, circuit: Any) -> str:
        pass

    @abstractmethod
    def from_qasm(self, qasm: str) -> Any:
        pass

    @staticmethod
    def get_translation(circuit: Circuit, language: str) -> Circuit:
        circuit_qasm_translator = Translation.object_map[circuit.__class__]
        qasm_circuit_translator = Translation.translation_map[language]
        qasm = circuit_qasm_translator.to_qasm(circuit)
        return qasm_circuit_translator.from_qasm(qasm)


class CirqTranslation(Translation):
    language: str = 'cirq'
    circuit_object: Type = cirq.Circuit

    def to_qasm(self, circuit: cirq.Circuit) -> str:
        return circuit.to_qasm()

    def from_qasm(self, qasm: str) -> cirq.Circuit:
        return circuit_from_qasm(qasm)


class QiskitTranslation(Translation):
    language: str = 'qiskit'
    circuit_object: Type = qiskit.QuantumCircuit

    def to_qasm(self, circuit: qiskit.QuantumCircuit) -> str:
        return qasm2.dumps(circuit)

    def from_qasm(self, qasm: str) -> qiskit.QuantumCircuit:
        return qiskit.QuantumCircuit.from_qasm_str(qasm)
