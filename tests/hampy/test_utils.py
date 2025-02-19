from quantum_launcher.hampy.utils import shift_affected_qubits
from quantum_launcher.hampy.object import Equation
from quantum_launcher.hampy.equations import one_in_n
from qiskit.quantum_info import Pauli, SparsePauliOp

def check_sparse_same(op1,op2):
    s1 = set(op1.label_iter())
    s2 = set(op2.label_iter())
    return s1 == s2

def test_shift_affected_qubits():    
    sample_eq = one_in_n([0,2,4],6)
    shift_target = one_in_n([1,3,5],6)
    
    shifted_eq = shift_affected_qubits(sample_eq,1)

    assert check_sparse_same(shifted_eq.hamiltonian,shift_target.hamiltonian)

    sample_eq = one_in_n([1],6)
    shift_target = one_in_n([3],6)

    shifted_eq = shift_affected_qubits(sample_eq,2)
    
    assert check_sparse_same(shifted_eq.hamiltonian,shift_target.hamiltonian)

    sample_eq = one_in_n([4,5],6)
    shift_target = one_in_n([0,1],6)

    shifted_eq = shift_affected_qubits(sample_eq,2)

    #Check if qubits are wrapped around
    assert check_sparse_same(shifted_eq.hamiltonian,shift_target.hamiltonian)
