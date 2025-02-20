from quantum_launcher.workflow.pilotjob_scheduler import JobManager
from quantum_launcher import QuantumLauncher, Result
from quantum_launcher.problems import MaxCut
from quantum_launcher.routines.qiskit_routines import QAOA, QiskitBackend
from quantum_launcher.routines.qiskit_routines.algorithms import EducatedGuess
import shutil
import glob
import pytest
# TODO: Make tests take shorter time to launch, and address event loop problem


@pytest.fixture(scope="function", autouse=True)
def clean_env():
    yield
    for path in glob.glob(".qcgpjm-client"):
        shutil.rmtree(path)
    for path in glob.glob(".qcgpjm-service-*"):
        shutil.rmtree(path)


def test_job_manager(tmp_path):
    manager = JobManager()
    assert isinstance(manager, JobManager)

    problem = MaxCut.from_preset('default')
    algorithm = QAOA(p=1)
    backend = QiskitBackend('local_simulator')

    manager.submit_many(problem, algorithm, backend, f'{tmp_path}/')
    for _ in range(len(manager.jobs)):
        job_id, status = manager.wait_for_a_job()
        assert isinstance(job_id, str)
        assert status != 'FAILED'
        results = manager.read_results(job_id)
        assert isinstance(results, Result)
    not_a_job = manager.wait_for_a_job()
    assert not_a_job is None


def test_educated_guess(tmp_path):
    """ Testing function for QATM """
    pr = MaxCut.from_preset('default')
    educated_guess = EducatedGuess(2, 3)
    educated_guess.output_initial = f'{tmp_path}/'
    educated_guess.output_interpolated = f'{tmp_path}/'
    educated_guess.output = f'{tmp_path}/'
    backend = QiskitBackend('local_simulator')
    launcher = QuantumLauncher(pr, educated_guess, backend)

    # inform = launcher.process(save_pickle=True)
    inform = launcher.run()
    assert isinstance(inform, Result)
