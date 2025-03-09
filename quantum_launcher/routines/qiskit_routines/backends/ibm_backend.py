""" Backend Class for Qiskit Launcher """
from typing import Literal

from qiskit.providers import BackendV1, BackendV2
from qiskit_algorithms.optimizers import SPSA
from qiskit_ibm_runtime import Estimator, Sampler
from qiskit_ibm_runtime import Session, Options

from quantum_launcher.routines.qiskit_routines.backends.qiskit_backend import QiskitBackend


class IBMBackend(QiskitBackend):
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
        self.session = session
        super().__init__(name, options, backendv1v2)

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
