"""Tests for CronScheduler, seed_defaults, next_run_from, human_schedule."""
from __future__ import annotations

import sys
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import pytest

from skill_hub.store import SkillStore
from skill_hub import cron as _cron
from skill_hub.cron import (
    CronScheduler,
    human_schedule,
    next_run_from,
    seed_defaults,
    register_handler,
    _HANDLERS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path):
    """Isolated SkillStore (creates cron_jobs table via migration)."""
    store = SkillStore(db_path=tmp_path / "skill_hub.db")
    yield str(tmp_path / "skill_hub.db")
    store.close()


@pytest.fixture(autouse=True)
def clear_handlers():
    """Ensure _HANDLERS is clean between tests."""
    old = dict(_HANDLERS)
    _HANDLERS.clear()
    yield
    _HANDLERS.clear()
    _HANDLERS.update(old)


# ---------------------------------------------------------------------------
# seed_defaults
# ---------------------------------------------------------------------------


def test_seed_defaults_populates_five_jobs(db_path):
    seed_defaults(db_path)
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT count(*) FROM cron_jobs").fetchone()[0]
    assert count == 5


def test_seed_defaults_idempotent(db_path):
    seed_defaults(db_path)
    seed_defaults(db_path)  # second call must be a no-op
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT count(*) FROM cron_jobs").fetchone()[0]
    assert count == 5


# ---------------------------------------------------------------------------
# next_run_from
# ---------------------------------------------------------------------------


def test_next_run_from_returns_future_datetime():
    base = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    nxt = next_run_from("0 2 * * *", base)
    assert nxt is not None
    assert nxt > base


def test_next_run_from_invalid_schedule():
    result = next_run_from("not-a-cron-expression")
    assert result is None


def test_next_run_from_every_15_minutes():
    base = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    nxt = next_run_from("*/15 * * * *", base)
    assert nxt is not None
    # Should be 12:15
    assert nxt.minute == 15


# ---------------------------------------------------------------------------
# human_schedule
# ---------------------------------------------------------------------------


def test_human_schedule_known_preset():
    assert human_schedule("0 2 * * *") == "daily at 2:00 AM"
    assert human_schedule("0 0 * * 0") == "weekly on Sunday at midnight"
    assert human_schedule("*/15 * * * *") == "every 15 minutes"


def test_human_schedule_unknown_returns_expression():
    expr = "30 8 * * 1-5"
    assert human_schedule(expr) == expr


# ---------------------------------------------------------------------------
# CronScheduler._maybe_run — skips job when next run is in the future
# ---------------------------------------------------------------------------


def test_maybe_run_skips_future_job(db_path):
    scheduler = CronScheduler(db_path)
    ran = []
    register_handler("my_cmd", lambda: ran.append(1))

    # last_run_at = just now, so next run should be in the future
    future_last = datetime.now(timezone.utc).isoformat()
    job = {
        "id": 1,
        "name": "test-job",
        "schedule": "0 2 * * *",
        "command": "my_cmd",
        "enabled": 1,
        "last_run_at": future_last,
    }
    scheduler._maybe_run(job)
    assert ran == [], "job must NOT run when next trigger is in the future"


def test_maybe_run_fires_overdue_job(db_path):
    scheduler = CronScheduler(db_path)
    ran = []
    register_handler("my_cmd", lambda: ran.append(1))

    # last_run_at far in the past — job is overdue
    job = {
        "id": 1,
        "name": "test-job",
        "schedule": "* * * * *",  # every minute
        "command": "my_cmd",
        "enabled": 1,
        "last_run_at": "2000-01-01T00:00:00",
    }
    scheduler._maybe_run(job)
    assert ran == [1], "overdue job must run"


# ---------------------------------------------------------------------------
# CronScheduler._run_job — calls handler and updates last_run_at
# ---------------------------------------------------------------------------


def test_run_job_calls_handler_and_updates_db(db_path):
    seed_defaults(db_path)
    scheduler = CronScheduler(db_path)

    called = []
    register_handler("my_test_cmd", lambda: called.append(True))

    # Insert a test job
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO cron_jobs(name, schedule, command, enabled)"
            " VALUES('unit-test-job','* * * * *','my_test_cmd',1)"
        )

    job = {"id": None, "name": "unit-test-job", "command": "my_test_cmd"}
    scheduler._run_job(job)

    assert called == [True]
    # Check that last_run_at was updated
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT last_run_at, last_status, run_count FROM cron_jobs WHERE name='unit-test-job'"
        ).fetchone()
    assert row is not None
    assert row[1] == "ok"
    assert row[2] == 1


def test_run_job_skips_unknown_command(db_path):
    scheduler = CronScheduler(db_path)
    # No handler registered — should not raise, should not update DB
    job = {"id": None, "name": "ghost-job", "command": "no_such_handler"}
    scheduler._run_job(job)  # must not raise


def test_run_job_records_error_on_handler_exception(db_path):
    seed_defaults(db_path)
    scheduler = CronScheduler(db_path)

    def boom():
        raise ValueError("intentional test error")

    register_handler("boom_cmd", boom)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO cron_jobs(name, schedule, command, enabled)"
            " VALUES('boom-job','* * * * *','boom_cmd',1)"
        )

    job = {"id": None, "name": "boom-job", "command": "boom_cmd"}
    scheduler._run_job(job)  # must not raise

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT last_status, last_error FROM cron_jobs WHERE name='boom-job'"
        ).fetchone()
    assert row[0] == "error"
    assert "intentional test error" in (row[1] or "")
