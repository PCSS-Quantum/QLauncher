"""DWave algorithms"""

from abc import ABC, abstractmethod
from typing import Any

from qlauncher.base import Algorithm, Result
from qlauncher.base.models import BQM
from qlauncher.exceptions import DependencyError
from qlauncher.routines.dwave.backends import BQMBackend

try:
    from dimod import Sampler, SampleSet
    from dwave.samplers import SimulatedAnnealingSampler, SteepestDescentSampler, TabuSampler
except ImportError as e:
    raise DependencyError(e, install_hint='dwave') from e


class DwaveSolver(Algorithm[BQM, BQMBackend], ABC):
    def __init__(self, chain_strength: int = 1, num_reads: int = 1000, **alg_kwargs) -> None:
        self.chain_strength = chain_strength
        self.num_reads = num_reads
        self.label: str = 'TBD_TBD'
        super().__init__(**alg_kwargs)

    def run(self, problem: BQM, backend: BQMBackend) -> Result:
        res = self._solve_bqm(problem.bqm, self._get_sampler(), **self.alg_kwargs)
        return self._construct_result(res)

    def _solve_bqm(self, bqm: Any, sampler: Sampler, **kwargs) -> SampleSet:
        return sampler.sample(bqm, num_reads=self.num_reads, label=self.label, chain_strength=self.chain_strength, **kwargs)

    @abstractmethod
    def _get_sampler(self) -> Sampler:
        pass

    def _construct_result(self, result: SampleSet) -> Result:
        distribution = {}
        energies = {}
        for value, energy, occ in zip(result.record.sample, result.record.energy, result.record.num_occurrences, strict=True):
            bitstring = ''.join(map(str, value))
            if bitstring in distribution:
                distribution[bitstring] += occ
                continue
            distribution[bitstring] = occ
            energies[bitstring] = energy

        return Result.from_counts_energies(distribution, energies, result)


class Tabu(DwaveSolver):
    """Tabu search simulator backend"""

    def _get_sampler(self) -> Sampler:
        return TabuSampler()


class SimulatedAnnealing(DwaveSolver):
    """Simulated annealing simulator backend"""

    def _get_sampler(self) -> Sampler:
        return SimulatedAnnealingSampler()


class SteepestDescent(DwaveSolver):
    """Steepest descent simulator backend"""

    def _get_sampler(self) -> Sampler:
        return SteepestDescentSampler()
