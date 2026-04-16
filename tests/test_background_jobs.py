"""Tests for background job queue — SkillStore methods + JobDispatcher.

Uses a tmp_path SkillStore (per-test temp directory) for isolation.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import pytest

from skill_hub.store import SkillStore
from skill_hub import background_jobs as bj
from skill_hub.background_jobs import JobDispatcher, get_dispatcher, register_handler


# ---------------------------------------------------------------------------
# Fixture: fresh in-memory SkillStore
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path):
    """Temporary on-disk SkillStore — isolated per test."""
    s = SkillStore(db_path=tmp_path / "skill_hub.db")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# enqueue_job / dequeue_job
# ---------------------------------------------------------------------------


def test_enqueue_returns_int_id(store):
    job_id = store.enqueue_job("noop", {})
    assert isinstance(job_id, int)
    assert job_id >= 1


def test_dequeue_returns_highest_priority_first(store):
    # Lower priority number = higher priority (ASC ordering)
    store.enqueue_job("low", {"n": 1}, priority=8)
    store.enqueue_job("high", {"n": 2}, priority=2)
    store.enqueue_job("mid", {"n": 3}, priority=5)

    row = store.dequeue_job()
    assert row is not None
    assert row["kind"] == "high"
    assert row["priority"] == 2


def test_dequeue_marks_job_running(store):
    store.enqueue_job("test", {}, priority=5)
    row = store.dequeue_job()
    assert row is not None

    # Check DB state after dequeue
    cur = store._conn.execute(
        "SELECT status, attempts, started_at FROM background_jobs WHERE id=?",
        (row["id"],),
    )
    db_row = cur.fetchone()
    assert db_row["status"] == "running"
    assert db_row["attempts"] == 1
    assert db_row["started_at"] is not None


def test_dequeue_returns_none_on_empty_queue(store):
    assert store.dequeue_job() is None


def test_dequeue_skips_non_pending(store):
    jid = store.enqueue_job("test", {})
    # Manually mark as done
    store._conn.execute("UPDATE background_jobs SET status='done' WHERE id=?", (jid,))
    store._conn.commit()
    assert store.dequeue_job() is None


# ---------------------------------------------------------------------------
# complete_job
# ---------------------------------------------------------------------------


def test_complete_job_marks_done(store):
    jid = store.enqueue_job("test", {})
    store.dequeue_job()  # transitions to running
    store.complete_job(jid, result={"items": 3}, worker="litellm")

    row = store._conn.execute(
        "SELECT status, worker_used, result, completed_at FROM background_jobs WHERE id=?",
        (jid,),
    ).fetchone()
    assert row["status"] == "done"
    assert row["worker_used"] == "litellm"
    assert json.loads(row["result"]) == {"items": 3}
    assert row["completed_at"] is not None


def test_complete_job_with_empty_result(store):
    jid = store.enqueue_job("test", {})
    store.dequeue_job()
    store.complete_job(jid)  # no result argument
    row = store._conn.execute(
        "SELECT status, result FROM background_jobs WHERE id=?", (jid,)
    ).fetchone()
    assert row["status"] == "done"
    assert json.loads(row["result"]) == {}


# ---------------------------------------------------------------------------
# fail_job
# ---------------------------------------------------------------------------


def test_fail_job_deferred_below_max_attempts(store):
    """attempts < max_attempts → deferred (not failed)."""
    jid = store.enqueue_job("fragile", {})
    # dequeue increments attempts to 1
    store.dequeue_job()
    store.fail_job(jid, error="network error", max_attempts=3)

    row = store._conn.execute(
        "SELECT status FROM background_jobs WHERE id=?", (jid,)
    ).fetchone()
    assert row["status"] == "deferred"


def test_fail_job_failed_at_max_attempts(store):
    """attempts >= max_attempts → failed."""
    jid = store.enqueue_job("fragile", {})
    # Manually bump attempts to max_attempts value
    store._conn.execute(
        "UPDATE background_jobs SET attempts=3 WHERE id=?", (jid,)
    )
    store._conn.commit()
    store.fail_job(jid, error="too many retries", max_attempts=3)

    row = store._conn.execute(
        "SELECT status, error FROM background_jobs WHERE id=?", (jid,)
    ).fetchone()
    assert row["status"] == "failed"
    assert row["error"] == "too many retries"


def test_fail_job_truncates_long_errors(store):
    jid = store.enqueue_job("job", {})
    long_error = "x" * 2000
    store.fail_job(jid, error=long_error, max_attempts=99)
    row = store._conn.execute(
        "SELECT error FROM background_jobs WHERE id=?", (jid,)
    ).fetchone()
    assert len(row["error"]) == 1000


# ---------------------------------------------------------------------------
# pending_job_count
# ---------------------------------------------------------------------------


def test_pending_job_count_zero_on_empty(store):
    assert store.pending_job_count() == 0


def test_pending_job_count_includes_pending_and_deferred(store):
    store.enqueue_job("a", {})
    jid2 = store.enqueue_job("b", {})
    # Mark one as deferred directly
    store._conn.execute(
        "UPDATE background_jobs SET status='deferred' WHERE id=?", (jid2,)
    )
    store._conn.commit()
    assert store.pending_job_count() == 2


def test_pending_job_count_excludes_done(store):
    jid = store.enqueue_job("x", {})
    store.dequeue_job()
    store.complete_job(jid)
    assert store.pending_job_count() == 0


def test_pending_job_count_excludes_failed(store):
    jid = store.enqueue_job("x", {})
    store._conn.execute(
        "UPDATE background_jobs SET status='failed' WHERE id=?", (jid,)
    )
    store._conn.commit()
    assert store.pending_job_count() == 0


# ---------------------------------------------------------------------------
# reset_deferred_jobs
# ---------------------------------------------------------------------------


def test_reset_deferred_jobs_moves_to_pending(store):
    jid = store.enqueue_job("a", {})
    store._conn.execute("UPDATE background_jobs SET status='deferred' WHERE id=?", (jid,))
    store._conn.commit()

    count = store.reset_deferred_jobs()
    assert count == 1
    row = store._conn.execute(
        "SELECT status FROM background_jobs WHERE id=?", (jid,)
    ).fetchone()
    assert row["status"] == "pending"


def test_reset_deferred_jobs_returns_rowcount(store):
    for _ in range(3):
        jid = store.enqueue_job("a", {})
        store._conn.execute(
            "UPDATE background_jobs SET status='deferred' WHERE id=?", (jid,)
        )
    store._conn.commit()
    assert store.reset_deferred_jobs() == 3


# ---------------------------------------------------------------------------
# JobDispatcher.dispatch_one — happy path
# ---------------------------------------------------------------------------


@pytest.fixture()
def dispatcher(store):
    """Fresh JobDispatcher backed by in-memory store."""
    return JobDispatcher(store=store)


def test_dispatch_one_runs_handler_and_marks_done(store, dispatcher):
    """dispatch_one() with a registered handler runs it and marks done."""
    results = []

    @register_handler("_test_dispatch_ok")
    def _handler(payload: dict) -> dict:
        results.append(payload)
        return {"processed": True}

    store.enqueue_job("_test_dispatch_ok", {"key": "val"})

    # Patch _run_with_worker to call handler directly (bypass worker selection)
    with patch.object(
        dispatcher,
        "_run_with_worker",
        side_effect=lambda kind, payload: bj._HANDLERS[kind](payload),
    ):
        ran = dispatcher.dispatch_one()

    assert ran is True
    assert results == [{"key": "val"}]

    row = store._conn.execute(
        "SELECT status FROM background_jobs"
    ).fetchone()
    assert row["status"] == "done"


def test_dispatch_one_returns_false_on_empty_queue(store, dispatcher):
    assert dispatcher.dispatch_one() is False


def test_dispatch_one_marks_deferred_when_handler_raises(store, dispatcher):
    """When _run_with_worker raises, job should be deferred (attempts=1 < max 3)."""
    store.enqueue_job("_test_failing", {})

    with patch.object(
        dispatcher,
        "_run_with_worker",
        side_effect=RuntimeError("intentional failure"),
    ):
        ran = dispatcher.dispatch_one()

    assert ran is True
    row = store._conn.execute(
        "SELECT status FROM background_jobs"
    ).fetchone()
    # attempts=1 after dequeue → fail_job default max_attempts=3 → deferred
    assert row["status"] == "deferred"


# ---------------------------------------------------------------------------
# JobDispatcher non-blocking re-entrancy guard
# ---------------------------------------------------------------------------


def test_dispatch_one_non_reentrant(store, dispatcher):
    """dispatch_one() returns False immediately if a dispatch is already running."""
    barrier = threading.Barrier(2)
    second_result: list[bool] = []

    def slow_run(kind: str, payload: dict) -> dict:
        barrier.wait(timeout=2)  # wait until test thread arrives
        return {"done": True}

    store.enqueue_job("_slow_job", {})
    dispatcher._run_with_worker = slow_run  # type: ignore[method-assign]

    t = threading.Thread(target=dispatcher.dispatch_one)
    t.start()

    # Allow background thread to enter dispatch and set _running=True
    time.sleep(0.05)

    # Second call should see _running=True and return False
    second_result.append(dispatcher.dispatch_one())

    barrier.wait(timeout=2)  # release background thread
    t.join(timeout=2)

    assert second_result == [False]


# ---------------------------------------------------------------------------
# get_dispatcher singleton
# ---------------------------------------------------------------------------


def test_get_dispatcher_returns_singleton():
    """get_dispatcher() always returns the same instance."""
    import skill_hub.background_jobs as _bj_mod
    orig = _bj_mod._DISPATCHER
    _bj_mod._DISPATCHER = None
    try:
        d1 = get_dispatcher()
        d2 = get_dispatcher()
        assert d1 is d2
    finally:
        _bj_mod._DISPATCHER = orig


def test_get_dispatcher_returns_jobdispatcher_instance():
    import skill_hub.background_jobs as _bj_mod
    orig = _bj_mod._DISPATCHER
    _bj_mod._DISPATCHER = None
    try:
        d = get_dispatcher()
        assert isinstance(d, JobDispatcher)
    finally:
        _bj_mod._DISPATCHER = orig


# ---------------------------------------------------------------------------
# pending_count / reset_deferred via dispatcher
# ---------------------------------------------------------------------------


def test_dispatcher_pending_count(store, dispatcher):
    store.enqueue_job("a", {})
    store.enqueue_job("b", {})
    assert dispatcher.pending_count() == 2


def test_dispatcher_reset_deferred(store, dispatcher):
    jid = store.enqueue_job("a", {})
    store._conn.execute(
        "UPDATE background_jobs SET status='deferred' WHERE id=?", (jid,)
    )
    store._conn.commit()
    n = dispatcher.reset_deferred()
    assert n == 1
    assert store.pending_job_count() == 1
