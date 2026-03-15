import time

import pytest

from qlauncher import QLauncher, Result
from qlauncher.routines.qiskit import FALQON, QiskitBackend
from qlauncher.workflow.local_scheduler import LocalJobManager
from tests.utils.multiprocessing import check_subprocesses_exit
from tests.utils.problem import get_hamiltonian


@check_subprocesses_exit()
def test_job_manager() -> None:
    manager = LocalJobManager()
    assert isinstance(manager, LocalJobManager)

    problem = get_hamiltonian()
    algorithm = FALQON(max_reps=1)
    backend = QiskitBackend('local_simulator')

    manager.submit(QLauncher(problem, algorithm, backend).run)
    for _ in range(len(manager.jobs)):
        job_id = manager.wait_for_a_job(timeout=60)  # job_id=None => wait ANY
        assert isinstance(job_id, str)
        results = manager.read_results(job_id)
        assert isinstance(results, Result)

    # unknown job_id
    with pytest.raises(KeyError):
        manager.wait_for_a_job('definitely_not_a_job_id')

    # wait any on empty manager should error
    empty = LocalJobManager()
    with pytest.raises(ValueError):
        empty.wait_for_a_job()

    manager.clean_up()


@check_subprocesses_exit()
def test_job_manager_cancel() -> None:
    manager = LocalJobManager()
    assert isinstance(manager, LocalJobManager)

    problem = get_hamiltonian()
    algorithm = FALQON(max_reps=1000000000)
    backend = QiskitBackend('local_simulator')

    job_id = manager.submit(QLauncher(problem, algorithm, backend).run)

    time.sleep(1)
    manager.cancel(job_id)
    job_id = manager.wait_for_a_job(job_id, timeout=60)
    assert job_id is not None

    manager.clean_up()


@check_subprocesses_exit()
def test_job_manager_cancel_on_gc():
    manager = LocalJobManager()
    assert isinstance(manager, LocalJobManager)

    manager.submit(lambda: time.sleep(1000))

    time.sleep(1)
    # celowo bez clean_up(): dekorator ma wykryć, czy procesy i tak się posprzątały
