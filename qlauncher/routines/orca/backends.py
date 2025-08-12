from qlauncher.base import Backend
from qlauncher.exceptions import DependencyError

try:
    from ptseries.algorithms.binary_solvers import BinaryBosonicSolver
except ImportError as e:
    raise DependencyError(e, install_hint='orca', private=True) from e


class OrcaBackend(Backend):
    """ local backend """

    def get_bbs(
        self,
        pb_dim: int,
        objective,
        input_state,
        **kwargs
    ) -> BinaryBosonicSolver:
        """ returns BBS algorithm """
        return BinaryBosonicSolver(
            pb_dim=pb_dim,
            objective=objective,
            input_state=input_state,
            **kwargs
        )

    def get_args(self):
        return {}
