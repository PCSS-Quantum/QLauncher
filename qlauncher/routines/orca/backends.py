from qlauncher.base import Backend
from qlauncher.exceptions import DependencyError

try:
    from ptseries.tbi import create_tbi
except ImportError as e:
    raise DependencyError(e, install_hint='orca', private=True) from e


class OrcaBackend(Backend):
    """ local backend """

    def get_tbi(self):
        """Returns tbi
        """
        return create_tbi()

    def get_args(self):
        return {}
