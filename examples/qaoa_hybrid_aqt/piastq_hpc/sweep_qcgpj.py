"""Batch sweep of run.py across a Slurm allocation via QCG-PilotJob. Launch: sbatch submit_sweep.sh."""

import itertools
import os
import sys

from qcg.pilotjob.api.job import Jobs
from qcg.pilotjob.api.manager import LocalManager

HERE = os.path.dirname(os.path.abspath(__file__))

SEEDS = [1, 2, 3, 4, 5]
BACKENDS = ['aqt_ideal', 'aqt_noise']   # add 'hardware' once an ARNICA token is configured


def main() -> None:
    os.makedirs(os.path.join(HERE, 'logs'), exist_ok=True)
    manager = LocalManager()
    jobs = Jobs()
    for backend, seed in itertools.product(BACKENDS, SEEDS):
        name = f'{backend}_s{seed}'
        jobs.add(
            name=name,
            exec=sys.executable,
            args=['run.py', '--backend', backend, '--seed', str(seed), '--n', '10'],
            stdout=f'logs/{name}.out',
            stderr=f'logs/{name}.err',
            numCores=1,
            wd=HERE,
        )
    ids = manager.submit(jobs)
    manager.wait4(ids)
    print(manager.info(ids))
    manager.finish()


if __name__ == '__main__':
    main()
