import time
import pytest

from qlauncher.workflow.local_scheduler import LocalJobManager
from qlauncher.launcher.aql.aql_task import ManagerBackedTask


# Helpers
def _run_mini_scheduler(manager: LocalJobManager, tasks: list[ManagerBackedTask], timeout: float = 5.0):
    """
    Minimal AQL-like scheduler:
    - submits ready tasks,
    - waits for any job to finish,
    - reads results and sets them into task objects.
    """
    start = time.time()
    job_to_task: dict[str, ManagerBackedTask] = {}

    # Loop until all tasks terminal
    while not all(t.done() for t in tasks):
        # submit newly-ready tasks
        for t in tasks:
            if t.is_ready() and t.job_id() is None and not t.cancelled():
                jid = t._submit(manager)
                job_to_task[jid] = t

        # If nothing is running and not all done -> deadlock (bad deps)
        if not job_to_task and not all(t.done() for t in tasks):
            raise RuntimeError('Deadlock: no runnable tasks, but not all tasks done. Check dependencies.')

        # Wait for any job
        remaining = timeout - (time.time() - start)
        if remaining <= 0:
            raise TimeoutError

        jid = manager.wait_for_a_job(None, timeout=remaining)
        assert jid is not None
        t = job_to_task.pop(jid)

        # Read and finalize
        try:
            res = manager.read_results(jid)
            t._set_result(res)
        except BaseException as e:
            t._set_exception(e)

    manager.clean_up()


# Tests
def test_manager_backed_task_basic_and_deps():
    m = LocalJobManager()

    def a():
        time.sleep(0.1)
        return 2

    def b(x):
        time.sleep(0.1)
        return x + 3

    def c(x, y):
        return x * y

    ta = ManagerBackedTask(a)
    tb = ManagerBackedTask(b, dependencies=[ta], pipe_dependencies=True)
    tc = ManagerBackedTask(c, dependencies=[ta, tb], pipe_dependencies=True)

    _run_mini_scheduler(m, [ta, tb, tc], timeout=20.0)

    assert ta.done() and tb.done() and tc.done()
    assert ta.result() == 2
    assert tb.result() == 5
    assert tc.result() == 10


def test_manager_backed_task_cancel_before_submit():
    m = LocalJobManager()

    def long():
        time.sleep(2)
        return 'ok'

    t = ManagerBackedTask(long)
    assert t.cancel() is True
    assert t.cancelled() is True
    assert t.done() is True
    assert t.result() is None

    m.clean_up()


def test_manager_backed_task_exception_propagates():
    m = LocalJobManager()

    def boom():
        raise RuntimeError('boom')

    t = ManagerBackedTask(boom)
    _run_mini_scheduler(m, [t], timeout=20.0)

    assert t.done()
    with pytest.raises(RuntimeError):
        _ = t.result()
