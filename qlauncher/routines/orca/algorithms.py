from typing import Literal
from collections.abc import Callable
import numpy as np

from qlauncher.base import Problem, Algorithm, Backend, Result
from qlauncher.exceptions import DependencyError
from qlauncher.routines.orca.backends import OrcaBackend

try:
    from ptseries.algorithms.binary_solvers import BinaryBosonicSolver
except ImportError as e:
    raise DependencyError(e, install_hint='orca', private=True) from e


class BBS(Algorithm):
    """
    Binary Bosonic Solver algorithm class.

    This class represents the Binary Bosonic Solver (BBS) algorithm. BBS is a quantum-inspired algorithm that
    solves optimization problems by mapping them onto a binary bosonic system. It uses a training process
    to find the optimal solution.

    Attributes:
    - learning_rate (float): The learning rate for the algorithm.
    - updates (int): The number of updates to perform during training.
    - tbi_loops (str): The type of TBI loops to use.
    - print_frequency (int): The frequency at which to print updates.
    - logger (Logger): The logger object for logging algorithm information.

    """
    _algorithm_format = 'qubo'

    def __init__(self, algorithm_format: Literal['qubo', 'fn'] = 'qubo',
                 input_state: list[int] | None = None,
                 n_samples: int = 100,
                 gradient_mode: str = "parameter-shift",
                 gradient_delta: float = np.pi / 6,
                 sampling_factor: int = 1,
                 learning_rate: float = 5e-2,
                 learning_rate_flip: float = 1e-1,
                 updates: int = 100):
        super().__init__()
        self._algorithm_format = algorithm_format
        self.bbs_params = {
            'n_samples': n_samples,
            'gradient_mode': gradient_mode,
            'gradient_delta': gradient_delta,
            'sampling_factor': sampling_factor,
        }
        self.training_params = {
            'learning_rate': learning_rate,
            'learning_rate_flip': learning_rate_flip,
            'updates': updates,
        }
        self.input_state = input_state

    def run(self, problem: Problem, backend: Backend, formatter: Callable[[Problem], np.ndarray]) -> Result:

        if not isinstance(backend, OrcaBackend):
            raise ValueError(f'{backend.__class__} is not supported by BBS algorithm, use OrcaBackend instead')
        objective = formatter(problem)

        # TODO: use offset somehow
        if not callable(objective):
            objective, offset = objective

        if self.input_state is None:
            if not callable(objective):
                self.input_state = [(i + 1) % 2 for i in range(len(objective))]
            else:
                raise ValueError('input_state needs to be provided if objective is a function (callable)')

        tbi = backend.get_tbi()
        bbs = BinaryBosonicSolver(pb_dim=len(self.input_state),
                                  objective=objective,
                                  input_state=self.input_state,
                                  tbi=tbi,
                                  **self.bbs_params)

        bbs.train(**self.training_params)

        return self.construct_results(bbs)

    def get_bitstring(self, result: list[float]) -> str:
        return ''.join(map(str, map(int, result)))

    def construct_results(self, solver: BinaryBosonicSolver) -> Result:
        # TODO: add support for distribution (probably with different logger)
        best_bitstring = ''.join(
            map(str, map(int, solver.config_min_encountered)))
        best_energy = solver.E_min_encountered
        most_common_bitstring = None
        most_common_bitstring_energy = None
        distribution = None
        energy = None
        num_of_samples = solver.n_samples
        average_energy = None
        energy_std = None
        #! Todo: instead of None attach relevant info from 'results'
        # results fail to pickle correctly btw
        return Result(best_bitstring, best_energy, most_common_bitstring,
                      most_common_bitstring_energy, distribution, energy,
                      num_of_samples, average_energy, energy_std, None)
