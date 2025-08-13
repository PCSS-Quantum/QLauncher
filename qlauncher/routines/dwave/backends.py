"""DWave backends"""

from abc import abstractmethod, ABC
from qlauncher.base import Backend


from qlauncher.exceptions import DependencyError

try:
    from dimod import Sampler
    from dwave.system import DWaveSampler, EmbeddingComposite
    from dwave.samplers import SimulatedAnnealingSampler, TabuSampler, SteepestDescentSampler
except ImportError as e:
    raise DependencyError(e, install_hint='dwave') from e


class BQMBackend(Backend, ABC):
    """Base dwave backend"""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.sampler = self._get_sampler()

    @abstractmethod
    def _get_sampler(self) -> Sampler:
        """Get a dimod sampler"""


class TabuBackend(BQMBackend):
    """Tabu search simulator backend"""

    def __init__(self) -> None:
        super().__init__("TabuBackend")

    def _get_sampler(self):
        return TabuSampler()


class SimulatedAnnealingBackend(BQMBackend):
    """Simulated annealing simulator backend"""

    def __init__(self) -> None:
        super().__init__("SimulatedAnnealingBackend")

    def _get_sampler(self) -> Sampler:
        return SimulatedAnnealingSampler()


class SteepestDescentBackend(BQMBackend):
    """Steepest descent simulator backend"""

    def __init__(self) -> None:
        super().__init__("SteepestDescentBackend")

    def _get_sampler(self) -> Sampler:
        return SteepestDescentSampler()


class DwaveBackend(BQMBackend):
    """Real Dwave device backend"""

    def __init__(self) -> None:
        super().__init__("DwaveDevice")

    def _get_sampler(self) -> Sampler:
        return EmbeddingComposite(DWaveSampler())
