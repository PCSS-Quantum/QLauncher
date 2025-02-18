from qiskit.quantum_info import Pauli, SparsePauliOp

def shift_paulis(op: SparsePauliOp, shift: int) -> SparsePauliOp:
    """
    For each Pauli in the operator, shifts the Pauli string by the given amount.
    i.e (shift = 1) IIII -> IIII, IZIZ -> ZIZI, etc. !Might be unwanted! ZIII -> IIIZ
    Keeps the coefficients the same.
    
    Args:
        op: Operator to shift
        shift: Amount to shift by
    
    Returns:
        SparsePauliOp: Operator with shifted Paulis
    """
    # This might be stupid, but I found no built-in way to do this
    if shift == 0:
        return op

    npaulis = []
    ncoeffs = []
    
    for p_string, coeff in op.label_iter():
        p_string = p_string[shift:] + p_string[:shift]
        npaulis.append(Pauli(data=p_string))
        ncoeffs.append(coeff)
        
    return SparsePauliOp(npaulis, coeffs=ncoeffs)