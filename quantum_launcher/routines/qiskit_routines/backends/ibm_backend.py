""" IBM backend class for Qiskit routines """
from typing import Literal

from quantum_launcher.routines.qiskit_routines.backends.qiskit_backend import QiskitBackend
from quantum_launcher.import_management import DependencyError
try:
    from qiskit.providers import BackendV1, BackendV2
    from qiskit_algorithms.optimizers import SPSA
    from qiskit_ibm_runtime import Estimator, Sampler
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
        name: Literal['local_simulator', 'backendv1v2_simulator', 'device'],
        options: Options = None,
        backendv1v2: BackendV1 | BackendV2 = None,
        session: Session | None = None,
    ) -> None:
        self.session = session
        super().__init__(name, options, backendv1v2)

    @property
    def setup(self) -> dict:
        return {
            'name': self.name,
            'session': self.session
        }

    def _set_primitives_on_backend_name(self) -> None:
        if self.name != 'device':
            super()._set_primitives_on_backend_name()
            return

        if self.session is None:
            raise AttributeError(
                'Please instantiate a session if using other backend than local')
        else:
            self.estimator = Estimator(mode=self.session, options=self.options)
            self.sampler = Sampler(mode=self.session, options=self.options)
            self.optimizer = SPSA()
