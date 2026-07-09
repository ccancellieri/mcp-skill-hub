"""Daemon-level WAL resilience (#139): busy_timeout + startup checkpoint.

A contended long writer (e.g. index_skills) must not make high-frequency
small writers fail instantly with "database is locked", and a stale/growing
WAL from a prior crash or checkpoint starvation must be reclaimed at startup
on a best-effort basis without ever blocking or crashing init.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from skill_hub.store import SkillStore


def test_busy_timeout_set_on_connect(tmp_path):
    store = SkillStore(db_path=tmp_path / "skill_hub.db")
    try:
        row = store._conn.execute("PRAGMA busy_timeout").fetchone()
        assert row[0] >= 30000
    finally:
        store.close()


def test_startup_checkpoint_runs_and_does_not_crash(tmp_path):
    """A fresh WAL-mode DB: the startup checkpoint pragma must execute
    without raising, and the store must be usable afterwards."""
    store = SkillStore(db_path=tmp_path / "skill_hub.db")
    try:
        # journal_mode is WAL (or the graceful fallback logged a warning) —
        # either way init must have completed and the connection must work.
        row = store._conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0].lower() in ("wal", "delete", "truncate", "memory")
        store._conn.execute("SELECT 1").fetchone()
    finally:
        store.close()


def test_startup_checkpoint_failure_does_not_crash_init(tmp_path, monkeypatch):
    """Simulate a checkpoint that cannot complete (e.g. another connection
    holds the WAL) — init must degrade gracefully, not raise."""
    import sqlite3

    class _FlakyConn(sqlite3.Connection):
        def execute(self, sql, *a, **k):
            if "wal_checkpoint" in sql:
                raise sqlite3.OperationalError("database is locked")
            return super().execute(sql, *a, **k)

    real_connect = sqlite3.connect

    def _patched_connect(*args, **kwargs):
        kwargs.setdefault("factory", _FlakyConn)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", _patched_connect)
    store = SkillStore(db_path=tmp_path / "skill_hub.db")
    try:
        store._conn.execute("SELECT 1").fetchone()
    finally:
        store.close()
