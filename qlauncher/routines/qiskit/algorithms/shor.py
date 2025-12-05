import numpy as np
from qlauncher.base import Algorithm, Result
from qlauncher.routines.qiskit.backends.qiskit_backend import QiskitBackend
from qlauncher.base.problem_like import ControlledModularMultiplierGates

from qiskit import QuantumCircuit, QuantumRegister

class Shor(Algorithm[ControlledModularMultiplierGates, QiskitBackend]):
    
    def __init__(self, n_shots, **alg_kwargs):
        super().__init__(**alg_kwargs)
        self.n_shots = n_shots
    
    def run(self, problem: ControlledModularMultiplierGates, backend: QiskitBackend) -> Result:
        n_qubits = 2 * np.ceil(np.log2(problem.modulo))
        phase_kickback = QuantumRegister(n_qubits)
        eigen_state = QuantumRegister(problem.num_qubits - 1)
        
        full_circuit = QuantumCircuit(phase_kickback, eigen_state)
        
        if problem.eigen_state_prep:
            full_circuit.x(eigen_state[-3]) # to be tested if it shouldnt be just 0
        else:
            full_circuit.append(problem.eigen_state_prep, eigen_state[:problem.eigen_state_prep.num_qubits])
            
        for qubit_ix in range(n_qubits):
            full_circuit.append(problem.gates[qubit_ix], (phase_kickback[qubit_ix], eigen_state[:problem.gates[qubit_ix].num_qubits]))
            
        full_circuit.h(phase_kickback)
        full_circuit.measure(phase_kickback)
        
        job = backend.sampler.run([full_circuit], self.n_shots) 
        
        result = job.result()[0]
        
        return result