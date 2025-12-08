from typing import Literal

from qlauncher.base import Algorithm, Backend, Problem, ProblemLike, Result
from qlauncher.launcher.qlauncher import QLauncher
from qlauncher.workflow import PilotJobManager, SlurmJobManager, Workflow, WorkflowManager

ManagerKind = Literal['local', 'pilotjob', 'slurm']


def run_with_manager(
	problem: Problem | ProblemLike,
	algorithm: Algorithm,
	backend: Backend,
	*,
	manager: ManagerKind = 'local',
	cores: int = 1,
	output_path: str | None = None,
	**manager_kwargs,
) -> Result:
	"""
	Runs a single QLauncher job using one of the supported manager backends.

	Args:
	    problem (Problem): Problem to be solved.
	    algorithm (Algorithm): Algorithm that will be executed.
	    backend (Backend): Backend on which the algorithm will run.
	    manager (ManagerKind, optional): Type of manager to use.
	        Supported values:
	            * ``"local"`` – run directly via :class:`QLauncher`,
	            * ``"pilotjob"`` – use pilotjob :class:`JobManager`,
	            * ``"slurm"`` – use :class:`SlurmJobManager`.
	        Defaults to ``"local"``.
	    cores (int, optional): Number of CPU cores per job (interpretation
	        depends on the selected manager). Defaults to 1.
	    output_path (str | None, optional): Path prefix for output files,
	        required when ``manager="pilotjob"``. Ignored for ``"local"`` and
	        ``"slurm"``. Defaults to ``None``.
	    **manager_kwargs: Additional keyword arguments passed directly to the
	        selected manager's constructor (e.g. ``slurm_options``,
	        ``env_setup`` for :class:`SlurmJobManager`).

	Raises:
	    ValueError: If ``output_path`` is not provided for
	        ``manager="pilotjob"`` or if an unknown manager type is given.

	Returns:
	    Result: Result object produced by the executed job.
	"""
	if manager == 'local':
		launcher = QLauncher(problem, algorithm, backend)
		return launcher.run()

	if manager == 'pilotjob':
		if output_path is None:
			raise ValueError("output_path is required for manager='pilotjob'")

		pj_manager = PilotJobManager(**manager_kwargs)
		job_id = pj_manager.submit(
			problem=problem,
			algorithm=algorithm,
			backend=backend,
			output_path=output_path,
			cores=cores,
		)
		pj_manager.wait_for_a_job(job_id)
		result = pj_manager.read_results(job_id)
		pj_manager.clean_up()
		return result

	if manager == 'slurm':
		slurm_manager = SlurmJobManager(**manager_kwargs)
		job_id = slurm_manager.submit(
			problem=problem,
			algorithm=algorithm,
			backend=backend,
			cores=cores,
		)
		slurm_manager.wait_for_a_job(job_id)
		result = slurm_manager.read_results(job_id)
		slurm_manager.clean_up()
		return result

	raise ValueError(f'Unknown manager type: {manager!r}')


def run_workflow_with_manager(
	wf_manager: WorkflowManager,
	problem: Problem | ProblemLike,
	backend: Backend,
	*,
	manager: ManagerKind = 'local',
	cores: int = 1,
	output_path: str | None = None,
	**manager_kwargs,
) -> Result:
	"""
	Runs a :class:`WorkflowManager`-defined workflow using a selected manager.

	Args:
	    wf_manager (WorkflowManager): Workflow manager that defines the
	        high-level workflow and can be converted to a :class:`Workflow`.
	    problem (Problem): Problem to be solved.
	    backend (Backend): Backend on which the workflow will run.
	    manager (ManagerKind, optional): Type of manager to use
	        (``"local"``, ``"pilotjob"``, or ``"slurm"``). Defaults to
	        ``"local"``.
	    cores (int, optional): Number of CPU cores per job. Interpretation
	        depends on the selected manager. Defaults to 1.
	    output_path (str | None, optional): Path prefix for output files.
	        Required if ``manager="pilotjob"``. Defaults to ``None``.
	    **manager_kwargs: Additional keyword arguments passed directly to the
	        selected manager's constructor (see :func:`run_with_manager`).

	Raises:
	    ValueError: Propagated from :func:`run_with_manager` if an invalid
	        manager type is given or ``output_path`` is missing for
	        ``"pilotjob"``.

	Returns:
	    Result: Result object produced by executing the workflow.
	"""
	workflow_algorithm: Workflow = wf_manager.to_workflow()
	return run_with_manager(
		problem=problem,
		algorithm=workflow_algorithm,
		backend=backend,
		manager=manager,
		cores=cores,
		output_path=output_path,
		**manager_kwargs,
	)


if __name__ == '__main__':
	from qlauncher.problems import MaxCut
	from qlauncher.routines.qiskit import QAOA, QiskitBackend

	problem = MaxCut.from_preset('default')
	algorithm = QAOA(p=3)
	backend = QiskitBackend('local_simulator')

	result = run_with_manager(
		problem=problem,
		algorithm=algorithm,
		backend=backend,
		cores=1,
		manager='slurm',
		slurm_options={'licenses': 'orca1:1'},
		env_setup=[
			'module load Python/python-3.11.0',
			'source ~/venv/bin/activate',
		],
	)
	print(result)
