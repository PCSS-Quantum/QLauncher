"""qiskit_aer implementation of QiskitBackend"""
from typing import Literal
from qiskit.providers import BackendV1, BackendV2
from qiskit_ibm_runtime import Options
from quantum_launcher.routines.qiskit_routines.backends.qiskit_backend import QiskitBackend
from quantum_launcher.import_management import DependencyError
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit.primitives import BackendSamplerV2, BackendEstimatorV2
except ImportError as e:
    raise DependencyError(e, 'qiskit') from e


def _set_sampler_auto_transpile_run(sampler: BackendSamplerV2):
    func = sampler.run

    def run_wrapper(pubs, *args, shots: int | None = None):
        newpubs = []
        for pub in pubs:
            if isinstance(pub, QuantumCircuit):
                pub = transpile(pub.decompose(), sampler._backend, optimization_level=0)
            elif isinstance(pub, tuple):
                pub = (transpile(pub[0].decompose(), sampler._backend, optimization_level=0), *pub[1:])
            newpubs.append(pub)
        return func(newpubs, *args, shots=shots)

    sampler.run = run_wrapper
    return sampler


def _set_estimator_auto_transpile_run(estimator: BackendEstimatorV2):
    func = estimator.run

    def run_wrapper(pubs, *args, precision: float | None = None):
        try:
            return func(pubs, *args, precision=precision)
        except Exception:
            newpubs = []
            for pub in pubs:
                if isinstance(pub, tuple):
                    pub = (transpile(pub[0].decompose(), estimator._backend, optimization_level=0), *pub[1:])
                newpubs.append(pub)

            return func(newpubs, *args, precision=precision)

    estimator.run = run_wrapper
    return estimator


class AerBackend(QiskitBackend):
    """
    QiskitBackend utilizing the qiskit_aer library. Runs local simulations only, utilizing CUDA capable gpus if available.
    """

    def __init__(
        self,
        name: Literal['local_simulator'] | Literal['backendv1v2_simulator'] | Literal['device'],
        options: Options | None = None,
        simulation_method: str = 'automatic',
        simulation_device: Literal['CPU', 'GPU'] = 'CPU',
        transpile: bool = False,
        backendv1v2: BackendV1 | BackendV2 | None = None
    ) -> None:
        self.method = simulation_method
        self.device = simulation_device
        self.transpile = transpile
        super().__init__(name, options, backendv1v2)

    def _set_primitives_on_backend_name(self):
        if self.name == 'local_simulator':
            self.simulator = AerSimulator(method=self.method, device=self.device)

            bs = BackendSamplerV2(backend=self.simulator)
            be = BackendEstimatorV2(backend=self.simulator)
            self.sampler = _set_sampler_auto_transpile_run(bs) if self.transpile else bs
            self.estimator = _set_estimator_auto_transpile_run(be) if self.transpile else be
            self.optimizer = COBYLA()
        elif self.name == 'backendv1v2_simulator':
            noise_model = NoiseModel.from_backend(self.backendv1v2)
            self.simulator = AerSimulator(method=self.method, device=self.device, noise_model=noise_model)

            bs = BackendSamplerV2(backend=self.simulator)
            be = BackendEstimatorV2(backend=self.simulator)
            self.sampler = _set_sampler_auto_transpile_run(bs) if self.transpile else bs
            self.estimator = _set_estimator_auto_transpile_run(be) if self.transpile else be
            self.optimizer = COBYLA()
        else:
            raise ValueError(f"Unsupported mode for this backend:'{self.name}'")

    def set_options(self, **fields):
        """Set additional options for the instance AerSimulator"""
        self.simulator.set_options(**fields)
