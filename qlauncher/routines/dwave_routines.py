from collections.abc import Callable

from qlauncher.base import Algorithm, Problem, Backend, Result
from qlauncher.exceptions import DependencyError
try:
    from dimod.binary.binary_quadratic_model import BinaryQuadraticModel
    from dimod import SampleSet
    from dwave.system import DWaveSampler, EmbeddingComposite
    from dwave.samplers import SimulatedAnnealingSampler, TabuSampler, SteepestDescentSampler
except ImportError as e:
    raise DependencyError(e, install_hint='dwave') from e


class DwaveSolver(Algorithm):
    _algorithm_format = 'bqm'

    def __init__(self, chain_strength=1, num_reads=1000, **alg_kwargs) -> None:
        self.chain_strength = chain_strength
        self.num_reads = num_reads
        self.label: str = 'TBD_TBD'
        super().__init__(**alg_kwargs)

    def run(self, problem: Problem, backend: Backend, formatter: Callable) -> Result:
        self.label = f'{problem.name}_{problem.instance_name}'

        bqm: BinaryQuadraticModel = formatter(problem)

        res = self._solve_bqm(bqm, backend.sampler, **self.alg_kwargs)
        return self._construct_result(res)

    def _solve_bqm(self, bqm, sampler, **kwargs):
        res = sampler.sample(
            bqm, num_reads=self.num_reads, label=self.label, chain_strength=self.chain_strength, **kwargs)
        return res

    def _construct_result(self, result: SampleSet) -> Result:
        distribution = {}
        energies = {}
        for (value, energy, occ) in zip(result.record.sample, result.record.energy, result.record.num_occurrences, strict=True):
            bitstring = ''.join(map(str, value))
            if bitstring in distribution:
                distribution[bitstring] += occ
                continue
            distribution[bitstring] = occ
            energies[bitstring] = energy

        return Result.from_distributions(distribution, energies, result)


class TabuBackend(Backend):
    def __init__(self, name: str = "TabuSampler", parameters: list = None) -> None:
        super().__init__(name, parameters)
        self.sampler = TabuSampler()


class SimulatedAnnealingBackend(Backend):
    def __init__(self, name: str = "SimulatedAnnealingSampler", parameters: list = None) -> None:
        super().__init__(name, parameters)
        self.sampler = SimulatedAnnealingSampler()


class SteepestDescentBackend(Backend):
    def __init__(self, name: str = 'SteepestDescentBackend', parameters: list | None = None) -> None:
        super().__init__(name, parameters)
        self.sampler = SteepestDescentSampler()


class DwaveBackend(Backend):
    def __init__(self, name: str = "DWaveSampler", parameters: list = None) -> None:
        super().__init__(name, parameters)
        self.sampler = EmbeddingComposite(DWaveSampler())


__all__ = ['DwaveSolver', 'TabuBackend',
           'DwaveBackend', 'SimulatedAnnealingBackend', 'SteepestDescentBackend']
