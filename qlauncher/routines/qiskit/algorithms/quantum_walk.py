import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
from qiskit.circuit.library import StatePreparation
from qiskit import QuantumCircuit
from qlauncher.base.base import Backend, Problem, Algorithm, Result
from collections.abc import Callable, Sequence
from typing import Any
from qlauncher.routines.qiskit.backends.qiskit_backend import QiskitBackend
from qlauncher.routines.cirq import CirqBackend
from qiskit import transpile
from qlauncher.problems.other.dtqw_1D import DTQW_1D
from qlauncher.base.base import _ProblemLike




class QW_Helpers:
    k_zero = np.array([1, 0])[:, np.newaxis]
    k_one = np.array([0, 1])[:, np.newaxis]
    b_zero = k_zero.transpose()
    b_one = k_one.transpose()

    def display_res_with_coin(self, data, begin, end, title, show=True):
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

    def display_res_without_coin(self, data, begin, end, title, show=True):
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

    def to_ket(self, number: np.uint16, no_qubits):
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

    def to_bra(self, number: np.uint16, no_qubits):
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

    def run(self, problem: DTQW_1D, backend: Backend) -> Result:
        if isinstance(backend, (QiskitBackend, CirqBackend)):
            sampler = backend.samplerV1
        else:
            raise ValueError(f"The accepted backends are QiskitBackend and CirqBackend, got {type(backend)}")
        
        self.__set_coin_type(problem.coin_type)
        qc = self.create_dtqw_circuit(problem.no_qubits, problem.no_steps, problem.position)
        qc = transpile(qc, backend, optimization_level=3)
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
    
    def __set_coin_type(self, coin_type: CoinType):
        self.coin_type = coin_type

    def __create_c_one_gate(self, quantum_circuit, first_qubit_id, target_qubit_id):
        if target_qubit_id == first_qubit_id:
            quantum_circuit.x(target_qubit_id)
        else:
            control_qubits = [i for i in range(first_qubit_id, target_qubit_id)]
            quantum_circuit.mcx(control_qubits, target_qubit_id)
        return quantum_circuit

    def __create_inc_One_gate(self, quantum_circuit, no_qubits, first_qubit_id):
        for i in range(first_qubit_id + 1, no_qubits + 1):
            target_qubit_id = first_qubit_id - i + no_qubits
            quantum_circuit = self.__create_c_one_gate(
                quantum_circuit, first_qubit_id, target_qubit_id
            )
        return quantum_circuit

    def __create_control_zero_gate(
        self, quantum_circuit, first_qubit_id, target_qubit_id
    ):
        if target_qubit_id == first_qubit_id:
            quantum_circuit.x(target_qubit_id)
        else:
            control_qubits = [i for i in range(first_qubit_id, target_qubit_id)]
            quantum_circuit.x(control_qubits)
            quantum_circuit.mcx(control_qubits, target_qubit_id)
            quantum_circuit.x(control_qubits)
        return quantum_circuit

    def __create_dec_One_gate(self, quantum_circuit, no_qubits, first_qubit_id):
        for i in range(first_qubit_id + 1, no_qubits + 1):
            target_qubit_id = first_qubit_id - i + no_qubits
            quantum_circuit = self.__create_control_zero_gate(
                quantum_circuit, first_qubit_id, target_qubit_id
            )
        return quantum_circuit
    
    def __apply_coin(self, quantum_circuit, coin_id):
        match self.coin_type:
            case CoinType.HADAMARD:
                return quantum_circuit.h(coin_id)

    def create_dtqw_circuit(self, no_qubits, steps, position):
        q_ids = [i for i in range(no_qubits)]
        quantum_circuit = QuantumCircuit(no_qubits)
        initial_state = [q for vector in position for q in vector]
        prep = StatePreparation(initial_state)
        quantum_circuit.append(prep, q_ids)

        qubits = [i for i in range(0, no_qubits - 1)]
        qubits = [no_qubits - 1, *qubits]
        empty_circ = QuantumCircuit(no_qubits - 1)
        inc_gate = (
            self.__create_inc_One_gate(empty_circ.copy(), no_qubits - 1, 0)
            .to_gate()
            .control(1)
        )
        dec_gate = (
            self.__create_dec_One_gate(empty_circ.copy(), no_qubits - 1, 0)
            .to_gate()
            .control(1)
        )

        for _ in range(steps):
            self.__apply_coin(quantum_circuit, no_qubits - 1)
            quantum_circuit.append(inc_gate, qubits)
            quantum_circuit.x(no_qubits - 1)
            quantum_circuit.append(dec_gate, qubits)
            quantum_circuit.x(no_qubits - 1)
        return quantum_circuit
    
    