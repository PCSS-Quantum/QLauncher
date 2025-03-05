""" Backend Class for Qiskit Launcher """
from overrides import override
from typing import Literal

from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_aqt_provider import AQTProvider
from qiskit_aqt_provider.primitives import AQTSampler, AQTEstimator
from qiskit.primitives import StatevectorEstimator as LocalEstimator
from qiskit.primitives import StatevectorSampler as LocalSampler
from qiskit.primitives import BackendSampler, BackendEstimator
from qiskit.primitives import Sampler as SamplerV1
from qiskit.primitives import Estimator as EstimatorV1
from qiskit.providers import BackendV1, BackendV2
from qiskit_algorithms.optimizers import COBYLA, SPSA, Optimizer
from qiskit_ibm_runtime import Estimator, Sampler
from qiskit_ibm_runtime import Session, Options

from quantum_launcher.base import Backend
from quantum_launcher.routines.qiskit_routines.v2_wrapper import SamplerV2Adapter


class QiskitBackend(Backend):
    """
    Base class for backends compatible with qiskit.

    Attributes:
        name (str): The name of the backend.        
        options (Options): The options for the backend.
        backendv1v2 (BackendV1 | BackendV2 | None, optional): Predefined backend to use with name 'backendv1v2_simulator'
        sampler (BaseSampler): The sampler used for sampling.
        estimator (LocalEstimator): The estimator used for estimation.
        optimizer (Optimizer): The optimizer used for optimization.


    """

    def __init__(self, name: Literal['local_simulator', 'backendv1v2_simulator', 'device'], options: Options = None, backendv1v2: BackendV1 | BackendV2 | None = None):
        super().__init__(name)
        self.options = options
        self.backendv1v2 = backendv1v2
        self.estimator: LocalEstimator = None
        self.optimizer: Optimizer = None
        self._samplerV1: SamplerV1 | None = None
        self._set_primitives_on_backend_name()

    @property
    def samplerV1(self) -> Sampler:
        if self._samplerV1 is None:
            self._samplerV1 = SamplerV2Adapter(self.sampler)
        return self._samplerV1

    def _set_primitives_on_backend_name(self):
        if self.name == 'local_simulator':
            self.estimator = LocalEstimator()
            self.sampler = LocalSampler()
            self.optimizer = COBYLA()
        elif self.name == 'backendv1v2_simulator':
            self.estimator = BackendEstimator(backend=self.backendv1v2)
            self.sampler = BackendSampler(backend=self.backendv1v2)
            self.optimizer = COBYLA()
        else:
            self.estimator = None
            self.sampler = None
            self.optimizer = None


class QiskitIBMBackend(QiskitBackend):
    """ 
    An extension of QiskitBackend providing support for IBM sessions.

    Attributes:
        session (Session): The session associated with the backend.
    """

    def __init__(
        self,
        name: Literal['local_simulator', 'backendv1v2_simulator', 'device'],
        options: Options = None,
        backendv1v2: BackendV1 | BackendV2 = None,
        session: Session = None,
    ) -> None:
        super().__init__(name, options, backendv1v2)
        self.session = session
        self._set_primitives_on_backend_name()

    @property
    def setup(self) -> dict:
        return {
            'name': self.name,
            'session': self.session
        }

    def _set_primitives_on_backend_name(self) -> None:
        super()._set_primitives_on_backend_name()
        if self.estimator is not None:
            return  # super() method set appropriate primitives

        if self.session is None:
            raise AttributeError(
                'Please instantiate a session if using other backend than local')
        else:
            self.estimator = Estimator(mode=self.session, options=self.options)
            self.sampler = Sampler(mode=self.session, options=self.options)
            self.optimizer = SPSA()


class AQTBackend(QiskitBackend):
    """
    An extension of QiskitBackend providing support for Alpine Quantum Technologies (AQT) devices.

    Attributes:
        token (str, optional): AQT token, used for authorization when using real device backends.
        dotenv_path (str,optional): Path to a .env file containing the AQT token. (recommended to use)

    Usage Example
    -------------
    ::

        backend = AQTBackend(token="valid_token", name='device')

    or
    ::

        backend = AQTBackend(dotenv_path="./.env", name='device')

    with a .env file:
    ::

        AQT_TOKEN=valid_token


    """

    def __init__(
        self,
        name: Literal['local_simulator', 'backendv1v2_simulator', 'device'],
        options: Options = None,
        backendv1v2: BackendV1 | BackendV2 = None,
        token: str = None,
        dotenv_path: str = None,
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
