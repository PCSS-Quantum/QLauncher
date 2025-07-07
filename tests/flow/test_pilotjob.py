import glob
import shutil
import pytest
from quantum_launcher.workflow.pilotjob_scheduler import JobManager
from quantum_launcher import QuantumLauncher, Result
from quantum_launcher.problems import EC
from quantum_launcher.routines.qiskit_routines import FALQON, QiskitBackend
from quantum_launcher.routines.qiskit_routines.algorithms import EducatedGuess
# TODO: address event loop problem (To @dsiera: what was the problem?)


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

    problem = EC.from_preset('micro')
    algorithm = FALQON(max_reps=1)
    backend = QiskitBackend('local_simulator')

    manager.submit(problem, algorithm, backend, f'{tmp_path}/')
    for _ in range(len(manager.jobs)):
        job_id, status = manager.wait_for_a_job(timeout=5)
        assert isinstance(job_id, str)
        assert status != 'FAILED'
        results = manager.read_results(job_id)
        assert isinstance(results, Result)
    with pytest.raises(ValueError):
        manager.wait_for_a_job('definitely_not_a_job_id')
    with pytest.raises(ValueError):
        manager.wait_for_a_job()
    manager.clean_up()


def test_educated_guess(tmp_path):
    """ Testing function for QATM """
    pr = EC.from_preset('micro')
    educated_guess = EducatedGuess(2, 2, max_job_batch_size=1)
    educated_guess.output_initial = f'{tmp_path}/'
    educated_guess.output_interpolated = f'{tmp_path}/'
    educated_guess.output = f'{tmp_path}/'
    backend = QiskitBackend('local_simulator')
    launcher = QuantumLauncher(pr, educated_guess, backend)

    # inform = launcher.process(save_pickle=True)
    inform = launcher.run()
    assert isinstance(inform, Result)
