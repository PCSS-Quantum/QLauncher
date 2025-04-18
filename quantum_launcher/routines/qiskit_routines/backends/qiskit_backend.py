""" Base backend class for Qiskit routines. """
from typing import Literal


from quantum_launcher.base import Backend
from quantum_launcher.routines.qiskit_routines.v2_wrapper import SamplerV2Adapter
from quantum_launcher.routines.qiskit_routines.backends.backend_utils import (
    set_estimator_auto_run_behavior, set_sampler_auto_run_behavior,
    AUTO_TRANSPILE_ESTIMATOR_TYPE, AUTO_TRANSPILE_SAMPLER_TYPE
)
from quantum_launcher.import_management import DependencyError
try:
    from qiskit.providers import BackendV1, BackendV2
    from qiskit.primitives import (
        BackendSamplerV2,
        BackendEstimatorV2,
        StatevectorEstimator,
        StatevectorSampler,
        Sampler
    )

    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_ibm_runtime import Options
except ImportError as e:
    raise DependencyError(e, install_hint='qiskit') from e


class QiskitBackend(Backend):
    """
    Base class for backends compatible with qiskit.

    Attributes:
        name (str): The name of the backend.
        options (Options | None, optional): The options for the backend. Defaults to None.
        backendv1v2 (BackendV1 | BackendV2 | None, optional): Predefined backend to use with name 'backendv1v2'. Defaults to None.
        sampler (BaseSamplerV2): The sampler used for sampling.
        estimator (BaseEstimatorV2): The estimator used for estimation.
        optimizer (Optimizer): The optimizer used for optimization.
    """

    def __init__(
        self,
        name: Literal['local_simulator', 'backendv1v2'] | str,
        options: Options | None = None,
        backendv1v2: BackendV1 | BackendV2 | None = None,
        auto_transpile: bool = False
    ) -> None:
        """
        Args:
            name (Literal[&#39;local_simulator&#39;, &#39;backendv1v2&#39;] | str): Name or mode of operation, 'backendv1v2' allows for using a specific backend simulator.
            options (Options | None, optional): Defaults to None.
            backendv1v2 (BackendV1 | BackendV2 | None, optional): Used with name 'backendv1v2', sampler and estimator will use it. Defaults to None.
            auto_transpile (bool, optional): Whether to automatically transpile cirquits to the sampler and estimator backends. Defaults to False.
        """
        super().__init__(name)
        self.options = options
        self.backendv1v2 = backendv1v2
        self._auto_transpile = auto_transpile
        self._auto_assign = False
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
        elif self.name == 'backendv1v2':
            self.estimator = BackendEstimatorV2(backend=self.backendv1v2)
            self.sampler = BackendSamplerV2(backend=self.backendv1v2)
            self.optimizer = COBYLA()
        else:
            raise ValueError(f"Unsupported mode for this backend:'{self.name}'")

        self._configure_auto_transpile()

    def _configure_auto_transpile(self):
        """
        Set auto transpilation if turned on, on estimator and sampler if compatible.
        """
        if isinstance(self.estimator, AUTO_TRANSPILE_ESTIMATOR_TYPE):
            self.estimator = set_estimator_auto_run_behavior(
                self.estimator, auto_transpile=self._auto_transpile, auto_assign=self._auto_assign)
        if isinstance(self.sampler, AUTO_TRANSPILE_SAMPLER_TYPE):
            self.sampler = set_sampler_auto_run_behavior(
                self.sampler, auto_transpile=self._auto_transpile, auto_assign=self._auto_assign)
