"""DWave algorithms"""

from qlauncher.base import Algorithm, Result
from qlauncher.base.problem_like import BQM
from qlauncher.exceptions import DependencyError
from qlauncher.routines.dwave.backends import BQMBackend

try:
	from dimod import SampleSet
except ImportError as e:
	raise DependencyError(e, install_hint='dwave') from e


class DwaveSolver(Algorithm[BQM, BQMBackend]):
	def __init__(self, chain_strength=1, num_reads=1000, **alg_kwargs) -> None:
		self.chain_strength = chain_strength
		self.num_reads = num_reads
		self.label: str = 'TBD_TBD'
		super().__init__(**alg_kwargs)

	def run(self, problem: BQM, backend: BQMBackend) -> Result:
		res = self._solve_bqm(problem.bqm, backend.sampler, **self.alg_kwargs)
		return self._construct_result(res)

	def _solve_bqm(self, bqm, sampler, **kwargs):
		return sampler.sample(bqm, num_reads=self.num_reads, label=self.label, chain_strength=self.chain_strength, **kwargs)

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
