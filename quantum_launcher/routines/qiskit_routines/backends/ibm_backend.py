""" IBM backend class for Qiskit routines """
from typing import Literal

from quantum_launcher.routines.qiskit_routines.backends.qiskit_backend import QiskitBackend
from quantum_launcher.import_management import DependencyError
try:
    from qiskit.providers import BackendV1, BackendV2
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_ibm_runtime import EstimatorV2, SamplerV2
    from qiskit_ibm_runtime import Session, Options
except ImportError as e:
    raise DependencyError(e, 'qiskit') from e


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
        session: Session | None = None,
        auto_transpile: bool = False,
    ) -> None:
        self.session = session
        super().__init__(name, options, backendv1v2, auto_transpile)

    @property
    def setup(self) -> dict:
        return {
            'name': self.name,
            'session': self.session
        }

    def _set_primitives_on_backend_name(self) -> None:
        if self.name == 'local_simulator':
            super()._set_primitives_on_backend_name()
            return
        self._auto_assign = True
        if self.name == 'backendv1v2':
            self.estimator = EstimatorV2(self.backendv1v2)
            self.sampler = SamplerV2(self.backendv1v2)
            self.optimizer = COBYLA()

        elif self.name == 'session':
            if self.session is None:
                raise AttributeError(
                    'Please instantiate a session if using other backend than local')
            else:
                self.estimator = EstimatorV2(mode=self.session, options=self.options)
                self.sampler = SamplerV2(mode=self.session, options=self.options)
                self.optimizer = COBYLA()

        else:
            raise ValueError(f"Unsupported mode for this backend:'{self.name}'")

        self._configure_auto_transpile()
