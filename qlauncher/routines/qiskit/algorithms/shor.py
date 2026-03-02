import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFTGate

from qlauncher.base import Algorithm, Result
from qlauncher.base.problem_like import ShorCircuit
from qlauncher.routines.qiskit.backends.qiskit_backend import QiskitBackend


class Shor(Algorithm[ShorCircuit, QiskitBackend]):
	def __init__(self, n_shots: int, **alg_kwargs):
		super().__init__(**alg_kwargs)
		self.n_shots = n_shots

	def run(self, problem: ShorCircuit, backend: QiskitBackend) -> Result:
		n_qubits = problem.n_phase_kickback_qubits
		phase_kickback = QuantumRegister(n_qubits, 'phase')
		eigen_state = QuantumRegister(problem.num_qubits - 1, 'eigen')
		classical_reg = ClassicalRegister(n_qubits, 'creg')

		qft = QFTGate(num_qubits=n_qubits)
		qft_i = qft.inverse()

		full_circuit = QuantumCircuit(phase_kickback, eigen_state, classical_reg)

		if not problem.eigen_state_prep:
			full_circuit.x(eigen_state[0])
		else:
			full_circuit.append(problem.eigen_state_prep, eigen_state[: problem.eigen_state_prep.num_qubits])

		full_circuit.h(phase_kickback)

		for qubit_ix in range(n_qubits):
			full_circuit.append(problem.gates[n_qubits - qubit_ix - 1], [phase_kickback[n_qubits - qubit_ix - 1]] + eigen_state[: problem.gates[qubit_ix].num_qubits])

		full_circuit.append(qft_i, phase_kickback)

		full_circuit.measure(phase_kickback, classical_reg)

		job = backend.sampler.run([full_circuit], shots=self.n_shots)

		return job.result()[0]
