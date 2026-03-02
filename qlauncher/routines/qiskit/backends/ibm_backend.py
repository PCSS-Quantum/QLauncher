"""IBM backend class for Qiskit routines"""

from typing import Literal

from qiskit.providers import BackendV1, BackendV2
from qiskit_ibm_runtime import EstimatorV2, Options, SamplerV2, Session

from qlauncher.routines.qiskit.backends.qiskit_backend import QiskitBackend
from qlauncher.routines.qiskit.mitigation_suppression.base import CircuitExecutionMethod


class IBMBackend(QiskitBackend):
	"""
	An extension of QiskitBackend providing support for IBM sessions.

	Attributes:
		session (Session | None, optional): The session to use with name 'device'.
	"""

	def __init__(
		self,
		name: Literal['local_simulator', 'backendv1v2', 'session'],
		options: Options | None = None,
		backendv1v2: BackendV1 | BackendV2 | None = None,
		auto_transpile_level: Literal[0, 1, 2, 3] | None = None,
		error_mitigation_strategy: CircuitExecutionMethod | None = None,
		session: Session | None = None,
	) -> None:
		self.session = session
		super().__init__(
			name,
			options=options,
			backendv1v2=backendv1v2,
			auto_transpile_level=auto_transpile_level,
			error_mitigation_strategy=error_mitigation_strategy,
		)

	@property
	def setup(self) -> dict:
		return {'name': self.name, 'session': self.session}

	def _set_primitives_on_backend_name(self) -> None:
		if self.name == 'local_simulator':
			super()._set_primitives_on_backend_name()
			return
		self._auto_assign = True
		if self.name == 'backendv1v2':
			if self.backendv1v2 is None:
				raise AttributeError('Please indicate a backend when in backendv1v2 mode.')
			self.estimator = EstimatorV2(self.backendv1v2)
			self.sampler = SamplerV2(self.backendv1v2)

		elif self.name == 'session':
			if self.session is None:
				raise AttributeError('Please indicate a session when in session mode.')
			self.estimator = EstimatorV2(mode=self.session, options=self.options)
			self.sampler = SamplerV2(mode=self.session, options=self.options)
		else:
			raise ValueError(
				' '.join([
					f"Unsupported mode for this backend:'{self.name}'."
					"Please use one of the following: ['local_simulator', 'backendv1v2', 'session']"
				])
			)

		self._configure_auto_behavior()
