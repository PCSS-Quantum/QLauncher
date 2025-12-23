"""AQT backend class for Qiskit routines"""

from typing import Literal

from overrides import override
from qiskit.primitives import BaseEstimatorV2, BaseSamplerV2
from qiskit.providers import BackendV1, BackendV2
from qiskit_ibm_runtime import Options

from qlauncher.exceptions import DependencyError
from qlauncher.routines.qiskit.adapters import (
	EstimatorV1ToEstimatorV2Adapter,
	SamplerV1ToSamplerV2Adapter,
	TranslatingSampler,
	TranslatingSamplerV1,
)
from qlauncher.routines.qiskit.backends.qiskit_backend import QiskitBackend
from qlauncher.routines.qiskit.mitigation_suppression.base import CircuitExecutionMethod

try:
	from qiskit_aqt_provider import AQTProvider
	from qiskit_aqt_provider.primitives import AQTEstimator, AQTSampler
except ImportError as e:
	raise DependencyError(e, install_hint='aqt') from e


class AQTBackend(QiskitBackend):
	"""
	An extension of QiskitBackend providing support for Alpine Quantum Technologies (AQT) devices.

	Attributes:
		token (str, optional): AQT token, used for authorization when using real device backends.
		dotenv_path (str,optional): (recommended) Path to a .env file containing the AQT token. If dotenv_path is not None, the token will be ignored and the token from the .env file will be used.

	Usage Example
	-------------
	::

	backend = AQTBackend(token='valid_token', name='device')

	or

	::

	backend = AQTBackend(dotenv_path='./.env', name='device')

	with a .env file:

	::

	AQT_TOKEN = valid_token

	"""

	sampler: BaseSamplerV2
	estimator: BaseEstimatorV2

	def __init__(
		self,
		name: Literal['local_simulator', 'backendv1v2', 'device'],
		options: Options | None = None,
		backendv1v2: BackendV1 | BackendV2 | None = None,
		auto_transpile_level: Literal[0, 1, 2, 3] | None = None,
		error_mitigation_strategy: CircuitExecutionMethod | None = None,
		token: str | None = None,
		direct_access_url: str | None = None,
		dotenv_path: str | None = None,
	) -> None:
		self._direct_access_url = direct_access_url
		if dotenv_path is None:
			self.provider = AQTProvider(token if token is not None else 'DEFAULT_TOKEN', load_dotenv=False)
		else:
			self.provider = AQTProvider(load_dotenv=True, dotenv_path=dotenv_path)
		super().__init__(
			name,
			options=options,
			backendv1v2=backendv1v2,
			auto_transpile_level=auto_transpile_level,
			error_mitigation_strategy=error_mitigation_strategy,
		)

	@override
	def _set_primitives_on_backend_name(self) -> None:
		if self.name == 'local_simulator':
			self.name = self.provider.backends(backend_type='offline_simulator', name=r'.*no_noise')[0].name
		elif self.name == 'backendv1v2':
			if self.backendv1v2 is None and self._direct_access_url:
				self.backendv1v2 = self.provider.get_direct_access_backend(self._direct_access_url)
			elif self.backendv1v2 is None:
				raise ValueError('Please indicate a backend when in backendv1v2 mode.')
		elif self.name == 'device':
			available_online_backends = self.provider.backends(backend_type='device')
			if len(available_online_backends) == 0:
				raise ValueError(f'No online backends available for token {self.provider.access_token[:5]}...')
			self.name = available_online_backends[0].name
		else:
			raise ValueError(
				' '.join([
					f"Unsupported mode for this backend:'{self.name}'."
					"Please use one of the following: ['local_simulator', 'backendv1v2', 'device']"
				])
			)

		if self.backendv1v2 is None:
			self.backendv1v2 = self.provider.get_backend(name=self.name)

		self._estimatorv1 = AQTEstimator(self.backendv1v2)
		self.estimator = EstimatorV1ToEstimatorV2Adapter(self._estimatorv1)
		self._samplerV1 = TranslatingSamplerV1(AQTSampler(self.backendv1v2), self.compatible_circuit)
		self.sampler = TranslatingSampler(SamplerV1ToSamplerV2Adapter(self._samplerV1), self.compatible_circuit)

		self._configure_auto_behavior()
