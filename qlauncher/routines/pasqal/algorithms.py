import numpy as np

from qoolqit.execution.utils import CompilerProfile

from qlauncher.base import Algorithm, Result
from qlauncher.base.problem_like import QUBO
from qlauncher.exceptions import DependencyError
from qlauncher.routines.pasqal.backends import PasqalBackend

try:
	from qoolqit import AnalogDevice, Drive, InteractionEmbedder, PiecewiseLinear, QuantumProgram, Ramp, Register
except ImportError as e:
	raise DependencyError(e, install_hint='pasqal', private=False) from e


class RydbergAnalogSolver(Algorithm[QUBO, PasqalBackend]):
	def __init__(self, **alg_kwargs) -> None:
		super().__init__(**alg_kwargs)

	def run(self, problem: QUBO, backend: PasqalBackend) -> Result:
		matrix = problem.matrix.astype(np.float64)
		matrix += matrix.T
		matrix /= 2.0
		matrix /= np.max(abs(matrix))

		embedder = InteractionEmbedder()
		embedded_graph = embedder.embed(matrix)

		register = Register.from_graph(embedded_graph)
		# ref -> https://pasqal-io.github.io/qoolqit/latest/tutorials/basic_qubo/
		# Defining the annealing parameters
		omega = np.median(matrix[matrix > 0].flatten())
		delta_i = -2.0 * omega
		delta_f = 2.0 * omega
		T = 52.0

		# Defining the annealing schedule
		wf_amp = PiecewiseLinear([T / 4, T / 2, T / 4], [0.0, omega, omega, 0.0])
		wf_det = Ramp(T, delta_i, delta_f)
		drive = Drive(amplitude=wf_amp, detuning=wf_det)

		# Writing the quantum program
		program = QuantumProgram(register, drive)
		program.compile_to(AnalogDevice(), profile=CompilerProfile.MAX_DURATION)

		results = backend.get_device().run(program)
		counter = results[0].final_bitstrings
		return Result.from_counts_energies(counter, dict.fromkeys(counter.keys(), 0))
