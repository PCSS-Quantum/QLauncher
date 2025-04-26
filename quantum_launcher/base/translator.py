""" Module providing Translation between different universal quantum computers """
from typing import Any
from abc import ABC, abstractmethod

import qiskit
from qiskit import qasm2
from qiskit.compiler import transpile
from qiskit.transpiler.passes import RemoveBarriers

from quantum_launcher.exceptions import DependencyError
try:
    import cirq
    from cirq.contrib.qasm_import.qasm import circuit_from_qasm
except ImportError as e:
    raise DependencyError(e, 'cirq') from e


class Translation(ABC):
    """ Translation layer for circuits written in different languages """
    basis_gates: list[str] = []
    language: str = 'None'
    circuit_object: type = type
    language_name_mapping: dict[str, "Translation"] = {}
    circuit_class_mapping: dict[type, "Translation"] = {}

    def __init_subclass__(cls):
        Translation.language_name_mapping[cls.language] = cls()
        Translation.circuit_class_mapping[cls.circuit_object] = cls()

    @abstractmethod
    def to_qasm(self, circuit: Any) -> str:
        """ Translation from given circuit into qasm (as a string) """

    @abstractmethod
    def from_qasm(self, qasm: str) -> Any:
        """ Translation given in qasm (as a string) circuit into language specific object """

    @staticmethod
    def get_translation(circuit: Any, language: str) -> Any:
        """ Transpiles circuit into given languages basis_gates, translates it to qasm, and from qasm into desired languages object. """
        circuit_qasm_translator = Translation.circuit_class_mapping[circuit.__class__]
        qasm_circuit_translator = Translation.language_name_mapping[language]
        if isinstance(circuit, qiskit.QuantumCircuit):
            transpiled_circuit = transpile_circuit(circuit, qasm_circuit_translator.basis_gates)
        else:
            transpiled_circuit = circuit  # Transpilation is usually not needed
        qasm = circuit_qasm_translator.to_qasm(transpiled_circuit)
        return qasm_circuit_translator.from_qasm(qasm)


class CirqTranslation(Translation):
    """ Translation class for Cirq """
    basis_gates = ['x', 'y', 'z', 'cx', 'h', 'rx', 'ry', 'rz']
    language: str = 'cirq'
    circuit_object = cirq.Circuit

    def to_qasm(self, circuit: cirq.Circuit) -> str:
        return circuit.to_qasm()

    def from_qasm(self, qasm: str) -> cirq.Circuit:
        return circuit_from_qasm(qasm)


class QiskitTranslation(Translation):
    """ Translation class for Qiskit """
    basis_gates = ['x', 'y', 'z', 'cx', 'h', 'rx', 'ry', 'rz', 'u']
    language: str = 'qiskit'
    circuit_object = qiskit.QuantumCircuit

    def to_qasm(self, circuit: qiskit.QuantumCircuit) -> str:
        return qasm2.dumps(circuit)

    def from_qasm(self, qasm: str) -> qiskit.QuantumCircuit:
        return qiskit.QuantumCircuit.from_qasm_str(qasm)


def transpile_circuit(qc: qiskit.QuantumCircuit, basis_gates: list[str]) -> qiskit.QuantumCircuit:
    """Makes circuit compatible with cirq

    Args:
        qc (qiskit.QuantumCircuit): circuit

    Returns:
        qiskit.QuantumCircuit: transpiled circuit
    """
    qc_transpiled = transpile(
        qc,
        basis_gates=basis_gates,
        optimization_level=3
    )

    return RemoveBarriers()(qc_transpiled)
