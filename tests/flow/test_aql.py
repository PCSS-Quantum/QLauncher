from __future__ import annotations

import inspect

from qlauncher import QLauncher, Result
from qlauncher.routines.qiskit import FALQON, QiskitBackend
from qlauncher.workflow.local_scheduler import LocalJobManager
from qlauncher.launcher.aql.aql import AQL

from tests.utils.multiprocessing import check_subprocesses_exit
from tests.utils.problem import get_hamiltonian


# Helpers
def _wait_aql_done(aql: AQL, timeout: float = 30.0) -> None:
	"""AQL uses a daemon scheduler thread; tests should ensure it finishes."""
	evt = getattr(aql, "_scheduler_done", None)
	assert evt is not None
	assert evt.wait(timeout), "AQL scheduler did not finish in time"


def _run_kwargs_that_wont_break(fn) -> dict:
	"""Return a minimal kwargs dict that is safe for real QLauncher.run.

	- If run(...) has a 'shots' parameter, we set a small value.
	- If run(...) does NOT accept **kwargs, we also add a deliberately-invalid
	  kwarg so AQL's filtering path is exercised.
	"""
	kwargs: dict = {}
	sig = inspect.signature(fn)
	accepts_varkw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())

	if "shots" in sig.parameters:
		kwargs["shots"] = 16
	elif "num_reads" in sig.parameters:
		kwargs["num_reads"] = 4

	if not accepts_varkw:
		kwargs["__invalid_should_be_filtered__"] = 1

	return kwargs


class SpyManager(LocalJobManager):
	"""LocalJobManager + telemetry for submission ordering."""

	def __init__(self):
		super().__init__()
		self.submit_calls: list[dict] = []

	def submit(self, function, **kwargs):
		jid = super().submit(function, **kwargs)
		name = getattr(function, "__name__", type(function).__name__)
		tag = kwargs.get("tag", None)

		if name == "_get_compatible_problem":
			label = f"format:{tag}" if tag is not None else "format"
		elif name == "_gateway_true":
			label = "gateway"
		elif name == "quantum_run":
			label = f"quantum:{tag}" if tag is not None else "quantum"
		else:
			label = f"classical:{tag}" if tag is not None else "classical"

		self.submit_calls.append({"jid": jid, "fn": name, "kwargs": dict(kwargs), "label": label})
		return jid


# Backends
class DeviceLikeQiskitBackend(QiskitBackend):
	"""Qiskit backend that behaves like a 'device' for AQL optimize_session,
	but still runs on local simulator."""
	def __init__(self, name: str):
		super().__init__(name)
		self.is_device = True



# Tests
@check_subprocesses_exit()
def test_aql_default_runs_real_qlaunchers_and_preserves_add_order() -> None:
	mgr = SpyManager()
	aql = AQL(mode="default", manager=mgr)

	problem = get_hamiltonian()
	backend = QiskitBackend("local_simulator")

	# Use slightly different configs to reduce the risk of identical results.
	algo1 = FALQON(max_reps=1)
	algo2 = FALQON(max_reps=2)

	# Build real launchers.
	ql1 = QLauncher(problem, algo1, backend)
	ql2 = QLauncher(problem, algo2, backend)

	# Feed only safe kwargs (and optionally an invalid one that should be filtered).
	kwargs1 = _run_kwargs_that_wont_break(ql1.run)
	kwargs2 = _run_kwargs_that_wont_break(ql2.run)

	# Add tasks in a known order.
	aql.add_task(ql1, manager_kwargs={"tag": "L1"}, **kwargs1)
	aql.add_task(ql2, manager_kwargs={"tag": "L2"}, **kwargs2)

	aql.start()
	results = aql.results(timeout=120)
	_wait_aql_done(aql)

	assert len(results) == 2
	assert isinstance(results[0], Result)
	assert isinstance(results[1], Result)

	# Submission order should follow add_task order in default mode.
	labels = [c["label"] for c in mgr.submit_calls]
	assert labels[0].startswith("classical:L1")
	assert labels[1].startswith("classical:L2")


@check_subprocesses_exit()
def test_aql_dependencies_prevent_early_submission_real_qlauncher() -> None:
	mgr = SpyManager()
	aql = AQL(mode="default", manager=mgr)

	problem = get_hamiltonian()
	backend = QiskitBackend("local_simulator")

	ql1 = QLauncher(problem, FALQON(max_reps=1), backend)
	ql2 = QLauncher(problem, FALQON(max_reps=1), backend)

	t1 = aql.add_task(ql1, manager_kwargs={"tag": "T1"}, **_run_kwargs_that_wont_break(ql1.run))
	aql.add_task(
		ql2,
		dependencies=[t1],
		manager_kwargs={"tag": "T2"},
		**_run_kwargs_that_wont_break(ql2.run),
	)

	aql.start()
	_ = aql.results(timeout=120)
	_wait_aql_done(aql)

	labels = [c["label"] for c in mgr.submit_calls]

	# T2 cannot be submitted before T1 finishes, so submission order must be T1 then T2.
	i1 = next(i for i, l in enumerate(labels) if l.startswith("classical:T1"))
	i2 = next(i for i, l in enumerate(labels) if l.startswith("classical:T2"))
	assert i2 > i1


@check_subprocesses_exit()
def test_aql_optimize_session_splits_device_task_with_real_qlauncher() -> None:
	mgr = SpyManager()
	aql = AQL(mode="optimize_session", manager=mgr)

	problem = get_hamiltonian()
	algo = FALQON(max_reps=1)
	backend = DeviceLikeQiskitBackend("local_simulator")

	ql = QLauncher(problem, algo, backend)
	aql.add_task(ql, manager_kwargs={"tag": "Q1"}, **_run_kwargs_that_wont_break(ql.run))

	aql.start()
	res = aql.results(timeout=120)[0]
	_wait_aql_done(aql)

	assert isinstance(res, Result)

	labels = [c["label"] for c in mgr.submit_calls]

	# optimize_session + "device" should submit at least: format, gateway, quantum, gateway
	assert any(l.startswith("format:Q1") for l in labels)
	assert any(l.startswith("quantum:Q1") for l in labels)
	assert labels.count("gateway") >= 2


@check_subprocesses_exit()
def test_aql_optimize_session_barrier_holds_unrelated_classical_until_after_quantum() -> None:
	mgr = SpyManager()
	aql = AQL(mode="optimize_session", manager=mgr)

	problem = get_hamiltonian()

	# Classical tasks: normal simulator backend
	backend_classical = QiskitBackend("local_simulator")
	# Quantum task: same simulator, but reported as device to exercise barrier.
	backend_device = DeviceLikeQiskitBackend("local_simulator")

	ql_pre = QLauncher(problem, FALQON(max_reps=1), backend_classical)
	ql_q = QLauncher(problem, FALQON(max_reps=1), backend_device)
	ql_after = QLauncher(problem, FALQON(max_reps=1), backend_classical)

	t_pre = aql.add_task(ql_pre, manager_kwargs={"tag": "PRE"}, **_run_kwargs_that_wont_break(ql_pre.run))
	aql.add_task(
		ql_q,
		dependencies=[t_pre],
		manager_kwargs={"tag": "Q"},
		**_run_kwargs_that_wont_break(ql_q.run),
	)
	aql.add_task(ql_after, manager_kwargs={"tag": "AFTER"}, **_run_kwargs_that_wont_break(ql_after.run))

	aql.start()
	_ = aql.results(timeout=180)
	_wait_aql_done(aql)

	labels = [c["label"] for c in mgr.submit_calls]

	q_idx = next(i for i, l in enumerate(labels) if l.startswith("quantum:Q"))
	after_idx = next(i for i, l in enumerate(labels) if l.startswith("classical:AFTER"))

	# Barrier property: unrelated classical task must not be submitted before quantum.
	assert after_idx > q_idx
	assert labels.count("gateway") >= 2
