""" Backend Class for Qiskit Launcher """
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from typing import Literal
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
    A class representing a local backend for Qiskit routines.

    This class extends the `Backend` and `QiskitRoutine` classes and provides functionality for a local backend.
    It allows for setting up a session, options, and various primitives such as estimators, samplers, and optimizers.

    Attributes:
        name (str): The name of the backend.
        session (Session): The session associated with the backend.
        options (Options): The options for the backend.
        primitive_strategy: The strategy for selecting primitives based on the backend name.
        sampler (BaseSampler): The sampler used for sampling.
        estimator (LocalEstimator): The estimator used for estimation.
        optimizer (Optimizer): The optimizer used for optimization.

    """

    def __init__(self, name: Literal['local_simulator', 'backendv1v2_simulator'], session: Session = None, options: Options = None, backendv1v2: BackendV1 | BackendV2 = None) -> None:
        super().__init__(name)
        self.session = session
        self.options = options
        self.backendv1v2 = backendv1v2
        self.primitive_strategy = None
        self.sampler = None
        self.estimator: LocalEstimator = None
        self.optimizer: Optimizer = None
        self._samplerV1: SamplerV1 | None = None
        self._set_primitives_on_backend_name()

    @property
    def setup(self) -> dict:
        return {
            'name': self.name,
            'session': self.session
        }

    @property
    def samplerV1(self) -> Sampler:
        if self._samplerV1 is None:
            self._samplerV1 = SamplerV2Adapter(self.sampler)
        return self._samplerV1

    def _set_primitives_on_backend_name(self) -> None:
        if self.name == 'local_simulator':
            self.estimator = LocalEstimator()
            self.sampler = LocalSampler()
            self.optimizer = COBYLA()
        elif self.name == 'backendv1v2_simulator':
            self.estimator = BackendEstimator(backend=self.backendv1v2)
            self.sampler = BackendSampler(backend=self.backendv1v2)
            self.optimizer = COBYLA()
        elif self.session is None:
            raise AttributeError(
                'Please instantiate a session if using other backend than local')
        else:
            self.estimator = Estimator(session=self.session, options=self.options)
            self.sampler = Sampler(session=self.session, options=self.options)
            self.optimizer = SPSA()


class AQTBackend(QiskitBackend):
    def __init__(self,
                 name: Literal['local_simulator', 'backendv1v2_simulator'],
                 options: Options = None,
                 token: str = None,
                 dotenv_path: str = None,
                 ) -> None:

        # TODO: This will probably need to be updated to handle custom backend urls, when we get our own computer
        if dotenv_path is None:
            self.provider = AQTProvider(token if token is not None else "DEFAULT_TOKEN", load_dotenv=False)
        else:
            self.provider = AQTProvider(load_dotenv=True, dotenv_path=dotenv_path)
        super().__init__(name, options=options)

    def _set_primitives_on_backend_name(self) -> None:
        if self.name == 'local_simulator':
            self.name = self.provider.backends(backend_type='offline_simulator', name=r".*no_noise")[0].name
        elif self.name == 'backendv1v2_simulator':
            available_online_backends = self.provider.backends(backend_type='device')
            if len(available_online_backends) == 0:
                raise ValueError(f"No online backends available for token {self.provider.access_token[:5]}...")
            self.name = available_online_backends[0].name

        backend = self.provider.get_backend(name=self.name)
        self.backendv1v2 = backend
        self.estimator = AQTEstimator(backend, options=self.options)
        self.sampler = AQTSampler(backend, options=self.options)
        self.optimizer = COBYLA()
