from functools import reduce
from typing import TypeAlias

from qiskit import QuantumCircuit

type_list: list[type] = [QuantumCircuit]
try:
	import cirq

	type_list.append(cirq.Circuit)
except ImportError:
	pass

CIRCUIT_FORMATS: TypeAlias = reduce(lambda a, b: a | b, type_list)
