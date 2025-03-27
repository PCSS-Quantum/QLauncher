""" Base backend class for Qiskit routines. """
from typing import Literal


from quantum_launcher.base import Backend
from quantum_launcher.routines.qiskit_routines.v2_wrapper import SamplerV2Adapter


from quantum_launcher.import_management import DependencyError
try:
    from qiskit.providers import BackendV1, BackendV2
    from qiskit.primitives import (BackendSamplerV2, BackendEstimatorV2, StatevectorEstimator,
                                   StatevectorSampler, Sampler, BaseSamplerV2, BaseEstimatorV2)

    from qiskit_algorithms.optimizers import COBYLA, Optimizer
    from qiskit_ibm_runtime import Options
except ImportError as e:
    raise DependencyError(e, install_hint='qiskit') from e


class QiskitBackend(Backend):
    """
    Base class for backends compatible with qiskit.

    Attributes:
        name (str): The name of the backend.
        options (Options | None, optional): The options for the backend. Defaults to None.
        backendv1v2 (BackendV1 | BackendV2 | None, optional): Predefined backend to use with name 'backendv1v2_simulator'. Defaults to None.
        sampler (BaseSamplerV2): The sampler used for sampling.
        estimator (BaseEstimatorV2): The estimator used for estimation.
        optimizer (Optimizer): The optimizer used for optimization.
    """

    def __init__(
        self,
        name: Literal['local_simulator', 'backendv1v2_simulator', 'device'],
        options: Options | None = None,
        backendv1v2: BackendV1 | BackendV2 | None = None
    ) -> None:
        super().__init__(name)
        self.options = options
        self.backendv1v2 = backendv1v2
        self._samplerV1: Sampler | None = None
        self._set_primitives_on_backend_name()

    @property
    def samplerV1(self) -> Sampler:
        if self._samplerV1 is None:
            self._samplerV1 = SamplerV2Adapter(self.sampler)
        return self._samplerV1

    def _set_primitives_on_backend_name(self):
        if self.name == 'local_simulator':
            self.estimator = StatevectorEstimator()
            self.sampler = StatevectorSampler()
            self.optimizer = COBYLA()
        elif self.name == 'backendv1v2_simulator':
            self.estimator = BackendEstimatorV2(backend=self.backendv1v2)
            self.sampler = BackendSamplerV2(backend=self.backendv1v2)
            self.optimizer = COBYLA()
        else:
            raise ValueError(f"Unsupported mode for this backend:'{self.name}'")
