import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate, Operation, Instruction
from qiskit._accelerate.circuit import CircuitInstruction as AccelerateInstruction
from qlauncher.base import Result
from qlauncher.routines.qiskit.backends.qiskit_backend import QiskitBackend
from qlauncher.utils import sum_counts
from .base import CircuitSamplingMethod


class PauliTwirling(CircuitSamplingMethod):
    def __init__(
        self,
        num_random_circuits: int,
        shots_per_circuit: int = 200,
        max_substitute_gates_per_circuit: int = 4,
        do_transpile: bool = True
    ) -> None:
        self.num_random_circuits = num_random_circuits
        self.shots_per_circuit = shots_per_circuit
        self.max_substitute_gates_per_circuit = max_substitute_gates_per_circuit
        self.do_transpile = do_transpile

    def _random_replacement_op(self, inst: AccelerateInstruction) -> list[AccelerateInstruction]:
        op: Operation = inst.operation
        match op.name:
            case "cx":
                return [
                    [
                        AccelerateInstruction(operation=Instruction(name='x', num_qubits=1,
                                              num_clbits=0, params=[]), qubits=[inst.qubits[0]]),
                        inst,
                        AccelerateInstruction(operation=Instruction(name='x', num_qubits=1,
                                              num_clbits=0, params=[]), qubits=[inst.qubits[0]]),
                        AccelerateInstruction(operation=Instruction(name='x', num_qubits=1,
                                              num_clbits=0, params=[]), qubits=[inst.qubits[1]]),
                    ],
                    [
                        AccelerateInstruction(operation=Instruction(name='z', num_qubits=1,
                                              num_clbits=0, params=[]), qubits=[inst.qubits[0]]),
                        inst,
                        AccelerateInstruction(operation=Instruction(name='z', num_qubits=1,
                                              num_clbits=0, params=[]), qubits=[inst.qubits[0]]),
                    ]][int(np.random.randint(0, 2))]

            case _:
                return [
                    inst
                ]

    def _twirl_circuit(self, transpiled_circuit: QuantumCircuit) -> QuantumCircuit:
        double_gates_with_indices: list[tuple[int, AccelerateInstruction]] = [
            (i, x) for i, x in enumerate(transpiled_circuit.data) if x.operation.num_qubits == 2]

        choice_idxs = np.random.choice(
            range(len(double_gates_with_indices)),
            size=min(
                self.max_substitute_gates_per_circuit,
                len(double_gates_with_indices)
            ),
            replace=False)

        data_cpy = [[x] for x in transpiled_circuit.data]

        for i in choice_idxs:
            data_cpy[i] = self._random_replacement_op(data_cpy[i][0])

        transpiled_circuit.data = sum(data_cpy, start=[])

        return transpiled_circuit

    def sample(self, circuit: QuantumCircuit, backend: QiskitBackend) -> Result:
        input_circ = transpile(circuit, backend.backendv1v2) if self.do_transpile else circuit
        results = backend.sampler.run(
            [
                transpile(self._twirl_circuit(input_circ), backend.backendv1v2) for _ in range(self.num_random_circuits)
            ],
            shots=self.shots_per_circuit).result()

        counts = [r.join_data().get_counts() for r in results]

        added = sum_counts(*counts)

        return Result.from_distributions(added, {k: -1 for k in added.keys()}, {})
