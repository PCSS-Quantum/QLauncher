import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation

from qlauncher.base.base import Algorithm, Backend, Result, _ProblemLike
from qlauncher.problems.other.dtqw_1D import DTQW_ND, CoinType
from qlauncher.routines.cirq import CirqBackend
from qlauncher.routines.qiskit.backends.qiskit_backend import QiskitBackend


class QW_Helper:
    k_zero = np.array([1, 0])[:, np.newaxis]
    k_one = np.array([0, 1])[:, np.newaxis]
    b_zero = k_zero.transpose()
    b_one = k_one.transpose()

    def display_res_with_coin(self, data: dict[str, float], begin: int, end: int, title: str, show: bool =True) -> None:
        new_data = {}
        for key, value in data.items():
            shortened_key = int(key[1:], 2)
            if shortened_key in new_data:
                new_data[shortened_key] += value
            else:
                new_data[shortened_key] = value
        x = []
        y = []
        for pos in range(begin, end):
            ys = 0.0
            if pos in new_data:
                ys = new_data[pos]
            y.append(ys)
            x.append(pos)
        plt.title(title)
        plt.xlabel("Position")
        plt.ylabel("No times walker was at position")
        plt.plot(x, y)
        if show:
            plt.show()

    def display_res_without_coin(self, data: dict[str, float], begin: int, end: int, title: str, show: bool =True) -> None:
        new_data = {}
        for key, value in data.items():
            shortened_key = int(key, 2)
            if shortened_key in new_data:
                new_data[shortened_key] += value
            else:
                new_data[shortened_key] = value
        x = []
        y = []
        for pos in range(begin, end):
            ys = 0.0
            if pos in new_data:
                ys = new_data[pos]
            y.append(ys)
            x.append(pos)
        plt.title(title)
        plt.xlabel("Position")
        plt.ylabel("No times walker was at position")
        plt.plot(x, y)
        if show:
            plt.show()

    def to_ket(self, number: int, no_qubits: int) -> np.ndarray:
        binary = None
        first = None
        if number >= 0:
            binary = (bin(number)[2:]).zfill(no_qubits)
            first = self.k_one if binary[0] == "1" else self.k_zero
        res = first
        for i in range(1, no_qubits):
            next_p = None
            next_p = self.k_one if binary[i] == "1" else self.k_zero
            res = np.kron(res, next_p)
        return res

    def to_bra(self, number: int, no_qubits: int) -> np.ndarray:
        first = None
        binary = (bin(number)[2:]).zfill(no_qubits)
        first = self.b_one if binary[0] == "1" else self.b_zero
        res = first
        for i in range(1, no_qubits):
            next_p = self.b_one if binary[i] == "1" else self.b_zero
            res = np.kron(res, next_p)
        return res

class DiscreteTimeQuantumWalk(Algorithm[_ProblemLike, QiskitBackend]):
    def __init__(self):
        self.algorithm_format = "DTQW"

    def run(self, problem: DTQW_ND, backend: Backend) -> Result:
        if isinstance(backend, (QiskitBackend, CirqBackend)):
            sampler = backend.samplerV1
        else:
            raise ValueError(f"The accepted backends are QiskitBackend and CirqBackend, got {type(backend)}")

        qc = self.build_dtqw_circuit(problem)
        qc.measure_all()
        result = sampler.run(qc).result()

        return Result(
            best_bitstring='',
            best_energy=1,
            most_common_bitstring='',
            most_common_bitstring_energy=0,
            distribution={},
            energies={},
            num_of_samples=0,
            average_energy=0,
            energy_std=0,
            result=result
        )

    def build_dtqw_circuit(self, problem) -> QuantumCircuit:
        no_qubits = sum(
            problem.no_coin_qubits_per_dim[d] + problem.no_pos_qubits_per_dim[d]
            for d in range(problem.no_dimensions)
        )

        q_ids = list(range(no_qubits))
        quantum_circuit = QuantumCircuit(no_qubits)
        qw_helper = QW_Helper()

        state = None

        for d in range(problem.no_dimensions):
            coin_init, pos_init = problem.initial_position_per_dim[d]

            coin_state = qw_helper.to_ket(
                coin_init,
                problem.no_coin_qubits_per_dim[d]
            )
            pos_state = qw_helper.to_ket(
                pos_init,
                problem.no_pos_qubits_per_dim[d]
            )

            dim_state = np.kron(coin_state, pos_state)

            state = dim_state if state is None else np.kron(state, dim_state)

        initial_state = [q for vector in state for q in vector]
        prep = StatePreparation(initial_state)
        quantum_circuit.append(prep, q_ids)

        max_steps = max(problem.no_steps_per_dim)
        for _ in range(max_steps):
            problem.apply_coin_op(quantum_circuit)
            problem.apply_shift_op(quantum_circuit, q_ids)

        return quantum_circuit

