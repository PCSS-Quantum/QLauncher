""" AQT backend class for Qiskit routines """
from typing import Literal
from overrides import override

from quantum_launcher.routines.qiskit_routines import QiskitBackend
from quantum_launcher.import_management import DependencyError

try:
    from qiskit.providers import BackendV1, BackendV2
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_ibm_runtime import Options
except ImportError as e:
    raise DependencyError(e, install_hint='qiskit') from e

try:
    from qiskit_aqt_provider import AQTProvider
    from qiskit_aqt_provider.primitives import AQTSampler, AQTEstimator
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

        AQT_TOKEN=valid_token

    """
    sampler: AQTSampler
    estimator: AQTEstimator

    def __init__(
        self,
        name: Literal['local_simulator', 'backendv1v2_simulator', 'device'],
        options: Options | None = None,
        backendv1v2: BackendV1 | BackendV2 | None = None,
        token: str | None = None,
        dotenv_path: str | None = None,
    ) -> None:

        # TODO: This will probably need to be updated to handle custom backend urls, when we get our own computer
        if dotenv_path is None:
            self.provider = AQTProvider(token if token is not None else "DEFAULT_TOKEN", load_dotenv=False)
        else:
            self.provider = AQTProvider(load_dotenv=True, dotenv_path=dotenv_path)
        super().__init__(name, options=options, backendv1v2=backendv1v2)

    @override
    def _set_primitives_on_backend_name(self) -> None:
        if self.name == 'local_simulator':
            self.name = self.provider.backends(backend_type='offline_simulator', name=r".*no_noise")[0].name
        elif self.name == 'backendv1v2_simulator':
            if self.backendv1v2 is None:
                raise ValueError("backendv1v2 should not be None when you plan on using it.")
        elif self.name == 'device':
            available_online_backends = self.provider.backends(backend_type='device')
            if len(available_online_backends) == 0:
                raise ValueError(f"No online backends available for token {self.provider.access_token[:5]}...")
            self.name = available_online_backends[0].name

        if self.backendv1v2 is None:
            self.backendv1v2 = self.provider.get_backend(name=self.name)

        self.estimator = AQTEstimator(self.backendv1v2, options=self.options)
        self.sampler = AQTSampler(self.backendv1v2, options=self.options)
        self.optimizer = COBYLA()
