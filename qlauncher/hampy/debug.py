""" Module with functionalities for debugging Hamiltonians and checking their boolean properties """
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp

from qlauncher.hampy.object import Equation


class TruthTable:
    """ TruthTable is class for debugging hamiltonians,
        it can plot the distribution and find edge energy values.
    """

    def __init__(self, equation: Equation | SparsePauliOp, return_int: bool = True):
        if isinstance(equation, SparsePauliOp):
            hamiltonian = equation
            self.size = hamiltonian.num_qubits
        if isinstance(equation, Equation):
            hamiltonian = equation.hamiltonian
            self.size = equation.size
        self.return_int = return_int
        self.truth_table = self._ham_to_truth(hamiltonian)
        self.lowest_value = min(self.truth_table.values())

    def count(self, value: float) -> int:
        """ Counts how many times energy value occurred """
        return sum(1 for i in self.truth_table.values() if i == value)

    def get_solutions(self, value: float) -> list[str]:
        """ Returns all bitstrings with energy equal to one specified by user.

        Args:
            value (float): energy value.

        Returns:
            list[str]: list of bitstrings
        """
        return list(filter(lambda x: self.truth_table[x] == value, self.truth_table.keys()))

    def count_min_value_solutions(self) -> int:
        """ Returns how number of optimal solutions

        Returns:
            int: number of solutions
        """
        return self.count(self.lowest_value)

    def get_min_value_solutions(self) -> list[str]:
        """ Returns list of optimal solutions.

        Returns:
            list[str]: list of solutions
        """
        return self.get_solutions(self.lowest_value)

    def check_if_binary(self) -> bool:
        """ Checks if the hamiltonian is binary (all energy values are either 0 or 1

        Returns:
            bool: true if all solutions have energy 0 or 1
        """
        return all((value in (0, 1)) for value in self.truth_table.values())

    def plot_distribution(self) -> None:
        """ Plots energy distribution.
        """
        values = list(self.truth_table.values())
        counts, bins = np.histogram(values, max(values) + 1)
        plt.stairs(counts, bins)
        plt.show()

    def _ham_to_truth(self, hamiltonian: SparsePauliOp):
        return {
            ''.join(reversed(bitstring)): value for bitstring, value in
            zip(
                product(('0', '1'), repeat=self.size),
                map(lambda x: int(x.real), hamiltonian.to_matrix().diagonal())
                if self.return_int else
                hamiltonian.to_matrix().diagonal()
            )
        }

    def __getitem__(self, index: str | int):
        if isinstance(index, int):
            index = bin(index)[2:].zfill(self.size)
        return self.truth_table[index]
