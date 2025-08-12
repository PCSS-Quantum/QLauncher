""" Launcher used for profiling (old version, waiting to be updated) """
import cProfile
import pstats
from qlauncher.base import Algorithm, Backend, Problem, Result
from .qlauncher import QLauncher


class ProfilingLauncher(QLauncher):
    """ Launcher made for debugging purposes of algorithms and other launchers focusing on performance issues """

    def __init__(self, problem: Problem, algorithm: Algorithm, backend: Backend, profiler_path: str = 'profiling-results.prof'):
        super().__init__(problem, algorithm, backend)
        self._profiler_path = profiler_path

    def run_profiling(self) -> Result:
        """ Run's the problem as usual in QLauncher, with profiling turned on.

        Returns:
            Result: Results
        """
        with cProfile.Profile() as pr:
            result = self.run()
        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.reverse_order()
        stats.print_stats()
        stats.dump_stats(self._profiler_path)
        return result
