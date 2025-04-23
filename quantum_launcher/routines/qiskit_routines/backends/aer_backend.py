"""qiskit_aer implementation of QiskitBackend"""
from typing import Literal
from quantum_launcher.routines.qiskit_routines.backends.qiskit_backend import QiskitBackend
from quantum_launcher.routines.qiskit_routines.backends.utils import (
    set_estimator_auto_run_behavior,
    set_sampler_auto_run_behavior
)
from quantum_launcher.import_management import DependencyError
try:
    from qiskit.primitives import BackendSamplerV2, BackendEstimatorV2
    from qiskit.providers import BackendV1, BackendV2

    from qiskit_algorithms.optimizers import COBYLA

    from qiskit_ibm_runtime import Options

    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
except ImportError as e:
    raise DependencyError(e, 'qiskit') from e


class AerBackend(QiskitBackend):
    """
    QiskitBackend utilizing the qiskit_aer library. Runs local simulations only, utilizing CUDA capable gpus if available.
    """

    def __init__(
        self,
        name: Literal['local_simulator', 'backendv1v2'],
        options: Options | None = None,
        backendv1v2: BackendV1 | BackendV2 | None = None,
        auto_transpile_level: int = -1,
        simulation_method: str = 'automatic',
        simulation_device: Literal['CPU', 'GPU'] = 'CPU',
    ) -> None:
        self.method = simulation_method
        self.device = simulation_device
        super().__init__(name, options, backendv1v2, auto_transpile_level)

    def _set_primitives_on_backend_name(self):
        if self.name == 'local_simulator':
            self.simulator = AerSimulator(method=self.method, device=self.device)
        elif self.name == 'backendv1v2':
            if self.backendv1v2 is None:
                raise AttributeError(
                    'Please indicate a backend when in backendv1v2 mode.')
            noise_model = NoiseModel.from_backend(self.backendv1v2)
            self.simulator = AerSimulator(method=self.method, device=self.device, noise_model=noise_model)
        else:
            raise ValueError(
                f"Unsupported mode for this backend:'{self.name}'. Please use one of the following: ['local_simulator', 'backendv1v2']")

        self.sampler = BackendSamplerV2(backend=self.simulator)
        self.estimator = BackendEstimatorV2(backend=self.simulator)
        self.optimizer = COBYLA()

        self._configure_auto_behavior()

    def set_options(self, **fields):
        """Set additional options for the instance AerSimulator"""
        self.simulator.set_options(**fields)
