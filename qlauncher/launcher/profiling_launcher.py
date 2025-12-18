import cProfile
import pstats
import subprocess
import sys
import tempfile
from pathlib import Path

from .qlauncher import QLauncher


class ProfilingLauncher(QLauncher):
	"""Launcher made for debugging purposes of algorithms and other launchers focusing on performance issues

	Attributes:
		profiler_path (str) path where to save the profiling results.

	"""

	def profile(
		self,
		profiler_path: str | Path | None,
		problem_search: bool = True,
		algorithm: bool = True,
	) -> pstats.Stats:
		if not problem_search:
			compatible_problem = self._get_compatible_problem()

		with cProfile.Profile() as pr:
			if problem_search:
				compatible_problem = self._get_compatible_problem()
			if algorithm:
				# compatible_problem is bounded as both scenarios have been covered
				self.algorithm.run(compatible_problem, self.backend)  # type: ignore

		stats = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE).reverse_order()
		if profiler_path is not None:
			stats.dump_stats(profiler_path)
		return stats

	def profile_with_visualization(self, stats: pstats.Stats) -> None:
		try:
			import snakeviz  # type: ignore # noqa: F401
		except ModuleNotFoundError as e:
			raise ModuleNotFoundError(
				'To use `ProfilingLauncher.profile_with_visualization` you need to instal snakeviz\n   Use `pip install snakeviz`'
			) from e
		with tempfile.NamedTemporaryFile() as tf:
			stats.dump_stats(tf.name)
			subprocess.run([sys.executable, '-m', 'snakeviz', tf.name])
