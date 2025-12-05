import numpy as np
from qlauncher.base import Algorithm, Result
from qlauncher.base.problem_like import GroverCircuit
from qlauncher.workflow.pilotjob_scheduler import JobManager
from qlauncher.routines.qiskit.backends.qiskit_backend import QiskitBackend

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import grover_operator

class Grover(Algorithm[GroverCircuit, QiskitBackend]):
    
    def __init__(self, n_shots, **alg_kwargs):
        super().__init__(**alg_kwargs)
        self.n_shots = n_shots
    
    def run(self, problem: GroverCircuit, backend: QiskitBackend) -> Result:
        
        grover_op = grover_operator(problem.oracle, problem.state_prep)
        full_circuit = problem.state_prep.copy()
        for _ in range(problem.num_iterations):
            full_circuit.compose(grover_op, inplace=True)
        full_circuit.measure_all()
        
        job = backend.sampler.run([full_circuit], self.n_shots) 
        
        result = job.result()[0]
        
        return result