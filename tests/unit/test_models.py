import numpy as np
from qiskit.quantum_info import SparsePauliOp

from qlauncher import models


def test_from_hamiltonian() -> None:
    """Testing translation from hamiltonian"""
    desired_matrix = np.array(
        [
            [-2.0, 4.0, 0.0, 0.0],
            [0.0, -4.0, 4.0, 0.0],
            [0.0, 0.0, -6.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    desired_offset = 4.0
    sparse_pauli = SparsePauliOp.from_sparse_list(
        [
            ('ZZ', [0, 1], 1),
            ('ZZ', [1, 2], 1),
            ('Z', [2], 2),
        ],
        num_qubits=4,
    )
    hamiltonian = models.Hamiltonian(sparse_pauli)
    qubo = hamiltonian.to_qubo()
    assert (qubo.matrix == desired_matrix).all()
    assert qubo.offset == desired_offset
