from qiskit import QuantumCircuit

from qlauncher.base import ProblemLike


class DTQW_1D(ProblemLike):

    def __init__(self, instance: list[set[int]] = None, instance_name: str = 'unnamed') -> None:
        super().__init__(instance=instance)
        self.instance_name = instance_name

        assert(len(instance) == 5)
        self.coin_type = instance[0]
        self.no_steps = instance[1]
        self.no_pos_qubits = instance[2]
        self.no_coin_qubits = instance[3]
        self.initial_position = instance[4]

    @property
    def setup(self) -> dict:
        return {
            'instance_name': self.instance_name
        }

    def _get_path(self) -> str:
        return f'{self.name}@{self.instance_name}'

    def apply_shift_op(self, quantum_circuit: QuantumCircuit, qubits: list[int]) -> None:
        empty_circ = QuantumCircuit(self.no_pos_qubits)
        inc_gate = (
             self.__create_inc_One_gate(empty_circ.copy(), self.no_pos_qubits, 0)
             .to_gate()
             .control(1)
            )
        dec_gate = (
            self.__create_dec_One_gate(empty_circ.copy(), self.no_pos_qubits, 0)
            .to_gate()
            .control(1)
            )

        quantum_circuit.append(inc_gate, qubits)
        quantum_circuit.x(self.no_pos_qubits)
        quantum_circuit.append(dec_gate, qubits)
        quantum_circuit.x(self.no_pos_qubits)

    def __create_c_one_gate(self, quantum_circuit: QuantumCircuit, first_qubit_id: int, target_qubit_id: int) -> None:
        if target_qubit_id == first_qubit_id:
            quantum_circuit.x(target_qubit_id)
        else:
            control_qubits = list(range(first_qubit_id, target_qubit_id))
            quantum_circuit.mcx(control_qubits, target_qubit_id)
        return quantum_circuit

    def __create_inc_One_gate(self, quantum_circuit: QuantumCircuit, no_qubits: int, first_qubit_id: int) -> QuantumCircuit:
        for i in range(first_qubit_id + 1, no_qubits + 1):
            target_qubit_id = first_qubit_id - i + no_qubits
            quantum_circuit = self.__create_c_one_gate(
                quantum_circuit, first_qubit_id, target_qubit_id
            )
        return quantum_circuit

    def __create_control_zero_gate(self, quantum_circuit: QuantumCircuit, first_qubit_id: int, target_qubit_id: int) -> QuantumCircuit:
        if target_qubit_id == first_qubit_id:
            quantum_circuit.x(target_qubit_id)
        else:
            control_qubits = list(range(first_qubit_id, target_qubit_id))
            quantum_circuit.x(control_qubits)
            quantum_circuit.mcx(control_qubits, target_qubit_id)
            quantum_circuit.x(control_qubits)
        return quantum_circuit

    def __create_dec_One_gate(self, quantum_circuit: QuantumCircuit, no_qubits: int, first_qubit_id: int) -> QuantumCircuit:
        for i in range(first_qubit_id + 1, no_qubits + 1):
            target_qubit_id = first_qubit_id - i + no_qubits
            quantum_circuit = self.__create_control_zero_gate(
                quantum_circuit, first_qubit_id, target_qubit_id
            )
        return quantum_circuit
