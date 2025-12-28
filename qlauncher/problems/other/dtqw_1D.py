from qiskit import QuantumCircuit
import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum

from qlauncher.base import ProblemLike

class CoinType(Enum):
    HADAMARD = 1

class DTQW_ND(ProblemLike):
    "Class for implementation problems of N compositions of 1 Dimensional Quantum Walk with discrete time"
    def __init__(self, instance: list[list[set[int]]] = None, instance_name: str = 'unnamed') -> None:

        self.instance_name = instance_name
        self.no_dimensions = len(instance)
        self.coin_types = []
        self.no_steps_per_dim = []
        self.no_pos_qubits_per_dim = []
        self.no_coin_qubits_per_dim = []
        self.initial_position_per_dim = []
        self.shunt_decomposition_dim = []
        for dimension in range(self.no_dimensions):

            self.coin_types += [instance[dimension][0]]
            self.no_steps_per_dim += [instance[dimension][1]]
            self.no_pos_qubits_per_dim += [instance[dimension][2]]
            self.no_coin_qubits_per_dim += [instance[dimension][3]]
            self.initial_position_per_dim += [instance[dimension][4]]
            self.shunt_decomposition_dim += [instance[dimension][5]]

    def apply_coin_op(self, quantum_circuit: QuantumCircuit) -> None:
        visited_qubits = 0
        for dimension in range(self.no_dimensions):
            visited_qubits += self.no_pos_qubits_per_dim[dimension]
            coin_ids = list(range(visited_qubits, visited_qubits + self.no_coin_qubits_per_dim[dimension]))
            match self.coin_types[dimension]:
                case CoinType.HADAMARD:
                    for coin_id in coin_ids:
                        quantum_circuit.h(coin_id)
            visited_qubits += self.no_coin_qubits_per_dim[dimension]

    def apply_shift_op(self, quantum_circuit: QuantumCircuit, qubits: list[int]) -> None:
        """
        Append shift operator (from Shunt Decomposition or original) into the quantum_circuit on 
        specified qubits.
        """
        visited_qubits = 0
        for dimension in range(self.no_dimensions):
            if self.shunt_decomposition_dim[dimension]:
                self.apply_decomposed_shift_op(quantum_circuit, qubits, dimension, visited_qubits)
            else:
                empty_circ = QuantumCircuit(self.no_pos_qubits_per_dim[dimension])
                inc_gate = (
                    self.__create_inc_One_gate(empty_circ.copy(), self.no_pos_qubits_per_dim[dimension], 0)
                    .to_gate(label ="Inc")
                    .control(self.no_coin_qubits_per_dim[dimension])
                    )
                dec_gate = (
                    self.__create_dec_One_gate(empty_circ.copy(), self.no_pos_qubits_per_dim[dimension], 0)
                    .to_gate(label = "Dec")
                    .control(self.no_coin_qubits_per_dim[dimension])
                    )
                " append() interpret first kubits as a control kubits, so we have to change the qubit list"
                qubits_ctrl_last_first = (
                    qubits[visited_qubits + self.no_pos_qubits_per_dim[dimension] :
                        visited_qubits + self.no_pos_qubits_per_dim[dimension] + self.no_coin_qubits_per_dim[dimension]]
                    +
                    qubits[visited_qubits :
                        visited_qubits + self.no_pos_qubits_per_dim[dimension]]
                )
                for i in range(self.no_coin_qubits_per_dim[dimension]):
                    quantum_circuit.x( visited_qubits + self.no_pos_qubits_per_dim[dimension] + i)
                quantum_circuit.append(inc_gate, qubits_ctrl_last_first)
                for i in range(self.no_coin_qubits_per_dim[dimension]):
                    quantum_circuit.x( visited_qubits + self.no_pos_qubits_per_dim[dimension] + i)
                quantum_circuit.append(dec_gate, qubits_ctrl_last_first)
            visited_qubits += self.no_pos_qubits_per_dim[dimension] + self.no_coin_qubits_per_dim[dimension]

    def apply_decomposed_shift_op(self, quantum_circuit: QuantumCircuit, qubits: list[int], dimension: int, visited_qubits: int) -> None:
        """
        Append shift operator from Shunt Decomposition into the quantum_circuit on specified qubits. Also return the resulting
        operator mostly for debugging purposes.
        """
        pos_n  = self.no_pos_qubits_per_dim[dimension]
        coin_n = self.no_coin_qubits_per_dim[dimension]

        pos_slice = qubits[
            visited_qubits :
            visited_qubits + pos_n
        ]

        coin_slice = qubits[
            visited_qubits + pos_n :
            visited_qubits + pos_n + coin_n
        ]

        qc = QuantumCircuit(pos_n + coin_n)

        local_all = list(range(pos_n + coin_n))

        self._conditional_complement_on_coin_zero(qc, local_all)
        self.__create_inc_One_gate(qc, pos_n, 0)
        self._conditional_complement_on_coin_zero(qc, local_all)


        quantum_circuit.compose(
            qc,
            qubits= pos_slice + coin_slice,
            inplace=True
        )

    def _conditional_complement_on_coin_zero(self, quantum_circuit: QuantumCircuit, qubits: list[int]) -> None:
        """
        """
        for i in range(len(qubits) - 1):
            quantum_circuit.cx(qubits[i + 1], qubits[i])

        for i in range(len(qubits) - 3, -1, -1):
            quantum_circuit.cx(qubits[i + 1], qubits[i])

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
        fig, axes = plt.subplots(
            self.no_dimensions, 1,
            figsize=(figsize[0], figsize[1] * self.no_dimensions),
            squeeze=False
        )

        for d in range(self.no_dimensions):
            ax = axes[d, 0]

            length = 2 ** self.no_pos_qubits_per_dim[d]
            start = self.initial_position_per_dim[d][1]

            G = nx.path_graph(length)
            pos = {i: (i, 0) for i in range(length)}

            colors = ["red" if i == start else "blue" for i in range(length)]

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
            ax.set_title(f"Dimension {d}")

        plt.tight_layout()
        plt.show()

