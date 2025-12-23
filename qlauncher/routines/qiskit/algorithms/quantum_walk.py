from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation

from qlauncher.base.base import Algorithm, Backend, Result, _ProblemLike
from qlauncher.problems.other.dtqw_1D import DTQW_1D
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


class CoinType(Enum):
    HADAMARD = 1


class DiscreteTimeQuantumWalk(Algorithm[_ProblemLike, QiskitBackend]):
    def __init__(self):
        self.coin_type = CoinType.HADAMARD
        self.algorithm_format = "dtqw_1d"

    def run(self, problem: DTQW_1D, backend: Backend) -> Result:
        if isinstance(backend, (QiskitBackend, CirqBackend)):
            sampler = backend.samplerV1
        else:
            raise ValueError(f"The accepted backends are QiskitBackend and CirqBackend, got {type(backend)}")

        self.__set_coin_type(problem.coin_type)
        qc = self.build_dtqw_circuit(problem)
        qc.measure_all()
        #qc = transpile(qc, backend, optimization_level=3)
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

    def build_dtqw_circuit(self, problem: DTQW_1D) -> QuantumCircuit:
        q_ids = list(range(problem.no_coin_qubits + problem.no_pos_qubits))
        quantum_circuit = QuantumCircuit(problem.no_coin_qubits + problem.no_pos_qubits)
        qw_helper = QW_Helper()

        coin_state = qw_helper.to_ket(problem.initial_position[0], problem.no_coin_qubits)
        walker_state = qw_helper.to_ket(problem.initial_position[1], problem.no_pos_qubits)

        initial_state = [q for vector in np.kron(coin_state,walker_state) for q in vector]
        prep = StatePreparation(initial_state)
        quantum_circuit.append(prep, q_ids)

        # qubits = list(range(0, problem.no_pos_qubits))
        # qubits = [problem.no_pos_qubits, *qubits]

        for _ in range(problem.no_steps):
            self.__apply_coin(quantum_circuit, problem.no_pos_qubits)
            self.__apply_shift(quantum_circuit, problem, q_ids)
        return quantum_circuit

    def __apply_coin(self, quantum_circuit: QuantumCircuit, coin_id: int) -> None:
        match self.coin_type:
            case CoinType.HADAMARD:
                quantum_circuit.h(coin_id)

    def __apply_shift(self, quantum_circuit: QuantumCircuit, problem: DTQW_1D, qubits: list[int]) -> None:
        problem.apply_shift_op(quantum_circuit, qubits)

    def __set_coin_type(self, coin_type: CoinType) -> None:
        self.coin_type = coin_type
