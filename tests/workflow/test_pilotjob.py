from quantum_launcher.workflow.pilotjob_scheduler import JobManager
from quantum_launcher import QuantumLauncher, Result
from quantum_launcher.problems import MaxCut
from quantum_launcher.routines.qiskit_routines import QAOA, QiskitBackend
from quantum_launcher.routines.qiskit_routines.algorithms import EducatedGuess


def test_job_manager():
    manager = JobManager()
    assert isinstance(manager, JobManager)

    problem = MaxCut.from_preset('default')
    algorithm = QAOA(p=1)
    backend = QiskitBackend('local_simulator')

    manager.submit_many(problem, algorithm, backend, 'output/')
    for _ in range(len(manager.jobs)):
        job_id, status = manager.wait_for_a_job()
        assert isinstance(job_id, str)
        assert status != 'FAILED'
        results = manager.read_results(job_id)
        assert isinstance(results, Result)
    not_a_job = manager.wait_for_a_job()
    assert not_a_job is None

# Takes too long
# def test_educated_guess():
#     """ Testing function for QATM """
#     pr = MaxCut.from_preset('default')
#     educated_guess = EducatedGuess(2, 3)
#     backend = QiskitBackend('local_simulator')
#     launcher = QuantumLauncher(pr, educated_guess, backend)

#     # inform = launcher.process(save_pickle=True)
#     inform = launcher.run()
#     educated_guess.manager.kill_manager_process()
#     assert isinstance(inform, Result)
