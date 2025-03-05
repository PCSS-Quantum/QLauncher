from typing import Literal

from qiskit.providers import BackendV1, BackendV2
from qiskit.primitives import StatevectorEstimator as LocalEstimator
from qiskit.primitives import StatevectorSampler as LocalSampler
from qiskit_algorithms.optimizers import COBYLA, Optimizer
from qiskit.primitives import BackendSampler, BackendEstimator
from qiskit.primitives import Sampler as SamplerV1

from qiskit_ibm_runtime import Options

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
    def samplerV1(self) -> SamplerV1:
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
