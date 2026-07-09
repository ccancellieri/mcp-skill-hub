"""Tests for scripts/drop_dead_ddl.py — the offline migration dropping the
model_rewards table and task claim columns left over after the #130 purge
(issue #131).
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import drop_dead_ddl  # noqa: E402


def _seed_pre_change_schema(db_path: Path) -> None:
    """Build a DB via SkillStore, then bolt the pre-#131 dead schema back on
    to simulate a database that predates this migration.
    """
    from skill_hub.store import SkillStore

    store = SkillStore(db_path=db_path)
    store.close()

    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS model_rewards (
            task_class   TEXT NOT NULL,
            domain       TEXT NOT NULL,
            tier         TEXT NOT NULL,
            trials       INTEGER NOT NULL DEFAULT 0,
            successes    REAL    NOT NULL DEFAULT 0.0,
            updated_at   TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (task_class, domain, tier)
        );
        """
    )
    for col in ("claimed_by", "claim_token", "claimed_at", "stealable_at"):
        conn.execute(f"ALTER TABLE tasks ADD COLUMN {col} TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tasks_claimed_by ON tasks(claimed_by)"
    )
    conn.commit()
    conn.close()


def _table_info(db_path: Path, table: str) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()


def _schema_names(db_path: Path, kind: str) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        return {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type=?", (kind,)
            )
        }
    finally:
        conn.close()


def test_run_drops_model_rewards_and_claim_columns(tmp_path: Path):
    db_path = tmp_path / "skill_hub.db"
    _seed_pre_change_schema(db_path)

    assert "model_rewards" in _schema_names(db_path, "table")
    assert "idx_tasks_claimed_by" in _schema_names(db_path, "index")
    assert {"claimed_by", "claim_token", "claimed_at", "stealable_at"} <= _table_info(
        db_path, "tasks"
    )

    drop_dead_ddl.run(db_path)

    assert "model_rewards" not in _schema_names(db_path, "table")
    assert "idx_tasks_claimed_by" not in _schema_names(db_path, "index")
    remaining_cols = _table_info(db_path, "tasks")
    assert not {"claimed_by", "claim_token", "claimed_at", "stealable_at"} & remaining_cols
    # Sibling columns survive untouched.
    assert "id" in remaining_cols
    assert "title" in remaining_cols


def test_run_is_idempotent_when_schema_already_dropped(tmp_path: Path):
    db_path = tmp_path / "skill_hub.db"
    _seed_pre_change_schema(db_path)

    drop_dead_ddl.run(db_path)
    # Second run against an already-migrated DB must not raise.
    drop_dead_ddl.run(db_path)

    assert "model_rewards" not in _schema_names(db_path, "table")


def test_run_against_current_schema_is_a_noop(tmp_path: Path):
    """A DB built from the current (post-#131) store schema has nothing to
    drop; the script must handle that cleanly."""
    from skill_hub.store import SkillStore

    db_path = tmp_path / "skill_hub.db"
    store = SkillStore(db_path=db_path)
    store.close()

    drop_dead_ddl.run(db_path)

    assert "model_rewards" not in _schema_names(db_path, "table")
    assert "id" in _table_info(db_path, "tasks")


def test_run_missing_db_raises(tmp_path: Path):
    with pytest.raises(SystemExit, match="no such database"):
        drop_dead_ddl.run(tmp_path / "does_not_exist.db")


def test_run_refuses_when_another_writer_holds_the_lock(tmp_path: Path):
    db_path = tmp_path / "skill_hub.db"
    _seed_pre_change_schema(db_path)

    blocker = sqlite3.connect(str(db_path))
    blocker.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(SystemExit, match="refusing to run"):
            drop_dead_ddl.run(db_path)
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()

    # Untouched — the refusal happened before any DDL ran.
    assert "model_rewards" in _schema_names(db_path, "table")


def test_run_with_vacuum_flag(tmp_path: Path):
    db_path = tmp_path / "skill_hub.db"
    _seed_pre_change_schema(db_path)

    drop_dead_ddl.run(db_path, vacuum=True)

    assert "model_rewards" not in _schema_names(db_path, "table")
