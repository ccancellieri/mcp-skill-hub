"""Acceptance tests for autopilot-lite (issue #21).

Covers the three acceptance criteria from the issue:

1. Synthetic claims queue drains.
2. SIGINT clean exit.
3. No-ruflo-dep verified by CI grep (lives in tests/test_no_ruflo_dep.py;
   this file double-checks the autopilot module specifically does not
   import claude_flow).
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC))

from skill_hub.autopilot import (
    AutopilotRunner,
    claim_next,
    insert_claim,
    list_claims,
    request_stop,
    run_autopilot,
)
from skill_hub.autopilot.claims import (
    ensure_schema,
    heartbeat,
    is_stop_requested,
    upsert_runner,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path: Path) -> Path:
    """Fresh SQLite DB for each test."""
    path = tmp_path / "autopilot.db"
    ensure_schema(path)
    return path


# ---------------------------------------------------------------------------
# Claims board basics
# ---------------------------------------------------------------------------

def test_insert_and_list_claims(db: Path):
    insert_claim(db, title="alpha", priority=5)
    insert_claim(db, title="beta", priority=1)
    rows = list_claims(db)
    assert {r.title for r in rows} == {"alpha", "beta"}


def test_claim_next_respects_priority_then_age(db: Path):
    # Lower priority number = more urgent.
    insert_claim(db, title="low-urgency", priority=9)
    insert_claim(db, title="high-urgency", priority=1)
    insert_claim(db, title="medium", priority=5)
    picked = claim_next(db, "runner-A")
    assert picked is not None
    assert picked.title == "high-urgency"
    assert picked.status == "running"
    assert picked.claimed_by == "runner-A"


def test_claim_next_skips_future_stealable_at(db: Path):
    future = datetime.now(timezone.utc) + timedelta(minutes=5)
    insert_claim(db, title="later", priority=1, stealable_at=future)
    insert_claim(db, title="now", priority=5)
    picked = claim_next(db, "r1")
    assert picked is not None
    assert picked.title == "now"
    # The future claim is still pending.
    assert claim_next(db, "r1") is None


def test_claim_next_returns_none_on_empty_queue(db: Path):
    assert claim_next(db, "r1") is None


# ---------------------------------------------------------------------------
# Acceptance #1 — synthetic queue drains
# ---------------------------------------------------------------------------

def test_synthetic_queue_drains(db: Path):
    """Insert 5 claims and confirm the autopilot loop drains them all."""
    titles = [f"job-{i}" for i in range(5)]
    for t in titles:
        insert_claim(db, title=t, priority=5, payload={"label": t})

    seen: list[str] = []

    def launcher(claim):
        seen.append(claim.title)

    result = run_autopilot(
        db,
        poll_interval=0.01,
        drain_and_exit=True,
        launcher=launcher,
    )

    assert result.drained == 5
    assert result.failed == 0
    assert set(seen) == set(titles)
    assert result.stopped_by == "empty"

    # Every claim ended in status='done'.
    final = list_claims(db)
    assert all(c.status == "done" for c in final), [
        (c.title, c.status, c.error) for c in final
    ]


def test_launcher_failure_marks_claim_failed_but_loop_continues(db: Path):
    insert_claim(db, title="will-fail", priority=1)
    insert_claim(db, title="will-pass", priority=5)

    def launcher(claim):
        if claim.title == "will-fail":
            raise RuntimeError("synthetic launcher error")

    result = run_autopilot(
        db,
        poll_interval=0.01,
        drain_and_exit=True,
        launcher=launcher,
    )
    assert result.drained == 2
    assert result.failed == 1
    final = {c.title: c for c in list_claims(db)}
    assert final["will-fail"].status == "failed"
    assert "synthetic launcher error" in (final["will-fail"].error or "")
    assert final["will-pass"].status == "done"


def test_max_claims_caps_drain(db: Path):
    for i in range(10):
        insert_claim(db, title=f"job-{i}", priority=5)
    result = run_autopilot(
        db,
        poll_interval=0.01,
        max_claims=3,
        drain_and_exit=False,
        launcher=lambda _c: None,
    )
    assert result.drained == 3
    assert result.stopped_by == "max_claims"
    pending = [c for c in list_claims(db) if c.status == "pending"]
    assert len(pending) == 7


# ---------------------------------------------------------------------------
# Acceptance #2 — SIGINT / autopilot_stop clean exit
# ---------------------------------------------------------------------------

def test_request_stop_exits_loop_cleanly(db: Path):
    """A continuously-polling loop must exit when autopilot_stop fires."""
    # No claims — the loop will sit in its poll-sleep until stopped.
    runner = AutopilotRunner(
        db,
        runner_id="stop-target",
        poll_interval=0.05,
        drain_and_exit=False,
    )
    result_box: dict = {}

    def target():
        result_box["r"] = runner.run()

    th = threading.Thread(target=target, daemon=True)
    th.start()
    # Give the runner a moment to register itself + start polling.
    time.sleep(0.1)

    # External stop signal (what autopilot_stop / SIGINT eventually do).
    runner.request_stop(reason="stop_requested")
    th.join(timeout=2.0)
    assert not th.is_alive(), "autopilot did not exit after request_stop"
    result = result_box["r"]
    assert result.stopped_by in {"stop_requested", "stop_flag"}
    assert result.drained == 0


def test_request_stop_via_sqlite_flag(db: Path):
    """A runner in another process picks up the stop signal from SQLite."""
    upsert_runner(db, "remote-runner")
    assert not is_stop_requested(db, "remote-runner")
    touched = request_stop(db, "remote-runner")
    assert touched == 1
    assert is_stop_requested(db, "remote-runner")

    # And running a loop with that runner_id exits immediately.
    result = run_autopilot(
        db,
        runner_id="remote-runner",
        poll_interval=10.0,  # large — would hang if the flag were ignored
        drain_and_exit=False,
        launcher=lambda _c: None,
    )
    assert result.stopped_by == "stop_requested"
    assert result.drained == 0


def test_request_stop_all_runners(db: Path):
    upsert_runner(db, "a")
    upsert_runner(db, "b")
    touched = request_stop(db, "")  # empty = all
    assert touched == 2
    assert is_stop_requested(db, "a")
    assert is_stop_requested(db, "b")


def test_keyboard_interrupt_in_launcher_stops_loop(db: Path):
    """A KeyboardInterrupt raised from inside a claim must end the loop."""
    insert_claim(db, title="boom", priority=1)
    insert_claim(db, title="never-runs", priority=5)

    def launcher(claim):
        if claim.title == "boom":
            raise KeyboardInterrupt()

    result = run_autopilot(
        db,
        poll_interval=0.01,
        drain_and_exit=True,
        launcher=launcher,
    )
    assert result.stopped_by == "signal"
    assert result.failed == 1
    titles_after = {c.title: c.status for c in list_claims(db)}
    # Second claim should still be pending — the loop exited before picking it up.
    assert titles_after["never-runs"] == "pending"
    assert titles_after["boom"] == "failed"


# ---------------------------------------------------------------------------
# Acceptance #3 — no ruflo dep in this module
# ---------------------------------------------------------------------------

def test_autopilot_module_does_not_import_claude_flow():
    import re
    root = Path(__file__).resolve().parent.parent.parent
    pattern = re.compile(r"^\s*(?:import\s+claude_flow|from\s+claude_flow)\b", re.MULTILINE)
    autopilot_dir = root / "src" / "skill_hub" / "autopilot"
    offenders = []
    for p in autopilot_dir.rglob("*.py"):
        if pattern.search(p.read_text(encoding="utf-8", errors="ignore")):
            offenders.append(str(p.relative_to(root)))
    assert not offenders, offenders


# ---------------------------------------------------------------------------
# Heartbeat / state housekeeping
# ---------------------------------------------------------------------------

def test_heartbeat_records_last_claim_id(db: Path):
    upsert_runner(db, "r1")
    cid = insert_claim(db, title="hb", priority=5)
    picked = claim_next(db, "r1")
    assert picked and picked.id == cid
    heartbeat(db, "r1", picked.id)
    import sqlite3
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT last_claim_id, last_heartbeat FROM autopilot_state WHERE runner_id=?",
            ("r1",),
        ).fetchone()
    assert row[0] == cid
    assert row[1] is not None
