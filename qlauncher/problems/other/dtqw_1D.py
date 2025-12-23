from qiskit import QuantumCircuit
import networkx as nx
import matplotlib.pyplot as plt

from qlauncher.base import ProblemLike


class DTQW_1D(ProblemLike):
    "Class for define problem for any Discrete Time Quantum Walk on one dimension"

    def __init__(self, instance: list[set[int]] = None, instance_name: str = 'unnamed') -> None:
        super().__init__(instance=instance)
        self.instance_name = instance_name

        assert(len(instance) == 6)
        self.coin_type = instance[0]
        self.no_steps = instance[1]
        self.no_pos_qubits = instance[2]
        self.no_coin_qubits = instance[3]
        self.initial_position = instance[4]
        self.shunt_decomposition = instance[5]

    @property
    def setup(self) -> dict:
        return {
            'instance_name': self.instance_name
        }

    def _get_path(self) -> str:
        return f'{self.name}@{self.instance_name}'

    def apply_shift_op(self, quantum_circuit: QuantumCircuit, qubits: list[int]) -> QuantumCircuit:
        """
        Append shift operator (from Shunt Decomposition or original) into the quantum_circuit on 
        specified qubits.
        """
        if self.shunt_decomposition:
            return self.apply_decomposed_shift_op(quantum_circuit, qubits)
        else:
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
            " append() interpret first kubit as a control kubit, so we change the qubit list"
            qubits_ctrl_last_first = [qubits[-1]] + qubits[:-1]
            quantum_circuit.x(self.no_pos_qubits)
            quantum_circuit.append(inc_gate, qubits_ctrl_last_first)
            quantum_circuit.x(self.no_pos_qubits)
            quantum_circuit.append(dec_gate, qubits_ctrl_last_first)
        return quantum_circuit

    def apply_decomposed_shift_op(self, quantum_circuit: QuantumCircuit, qubits: list[int]) -> QuantumCircuit:
        """
        Append shift operator from Shunt Decomposition into the quantum_circuit on specified qubits. Also return the resulting
        operator mostly for debugging purposes.
        """
        pos_qubits = list(range(self.no_pos_qubits))
        qc = QuantumCircuit(self.no_pos_qubits + self.no_coin_qubits)
        self._conditional_complement_on_coin_zero(qc, pos_qubits)
        self.__create_inc_One_gate(qc, self.no_pos_qubits, 0)
        self._conditional_complement_on_coin_zero(qc, pos_qubits)
        quantum_circuit.append(qc, qubits)
        return qc

    def _conditional_complement_on_coin_zero(self, quantum_circuit: QuantumCircuit, qubits: list[int]) -> None:
        """
        """
        for q_id, q in enumerate(qubits):
            quantum_circuit.cx(q_id + 1, q)
        for q_id, q in reversed(list(enumerate(qubits[:-1]))):
            quantum_circuit.cx(qubits[q_id + 1], q)

    def __create_c_one_gate(self, quantum_circuit: QuantumCircuit, first_qubit_id: int, target_qubit_id: int) -> None:
        """
        Append the multicontrol by one gate into the quantum circuit (or X gate if is just one qubit)
        """
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

    def visualize(self, figsize=(12, 3)) -> None:
        """
        Function for visualization the line
        """
        length = 2**self.no_pos_qubits
        start = self.initial_position[1]

        G = nx.path_graph(length)
        pos = {i: (i, 0) for i in range(length)}

        colors = ["red" if i == start else "blue" for i in range(length)]

        fig, ax = plt.subplots(figsize=figsize)

        nx.draw_networkx_edges(G, pos, width=2, ax=ax)
        nx.draw_networkx_nodes(
            G, pos,
            node_shape="o",
            node_size=1200,
            node_color=colors,
            edgecolors="black",
            linewidths=2,
            ax=ax
        )
        nx.draw_networkx_labels(G, pos, ax=ax)

        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.set_xlim(-0.5, length - 0.5)
        ax.set_ylim(-1.2, 1.2)
        plt.show()

