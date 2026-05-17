"""Minimal claims board — SQLite-backed work queue for autopilot-lite.

The schema is intentionally self-contained: autopilot owns the ``claims`` and
``autopilot_state`` tables and creates them on first use. This keeps the
``store.py`` migration block untouched while #20 (swarm-lite) decides on the
final claims contract.

Columns:
    id             integer primary key
    title          short label (worktree slug, issue title, ...)
    payload        free-form JSON the launcher consumes (e.g. issue url)
    priority       lower = more urgent (5 = default)
    status         pending | running | done | failed
    claimed_by     identifier of the runner that owns the claim (NULL = free)
    stealable_at   ISO timestamp — claim is invisible to pollers before this
    created_at     ISO timestamp (set on insert)
    started_at     ISO timestamp (set when claimed)
    finished_at    ISO timestamp (set on done/failed)
    error          short error string when status='failed'

The ``stealable_at`` column matches the term used in the issue body and lets
callers stagger claims (e.g. schedule one for "in 5 minutes" so it cannot be
picked up before then). NULL stealable_at means "immediately stealable".
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS claims (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    title          TEXT NOT NULL,
    payload        TEXT NOT NULL DEFAULT '{}',
    priority       INTEGER NOT NULL DEFAULT 5,
    status         TEXT NOT NULL DEFAULT 'pending'
                       CHECK (status IN ('pending','running','done','failed')),
    claimed_by     TEXT,
    stealable_at   TEXT,
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    started_at     TEXT,
    finished_at    TEXT,
    error          TEXT
);

CREATE INDEX IF NOT EXISTS idx_claims_status_priority
    ON claims(status, priority, created_at);

CREATE TABLE IF NOT EXISTS autopilot_state (
    runner_id        TEXT PRIMARY KEY,
    stop_requested   INTEGER NOT NULL DEFAULT 0,
    started_at       TEXT NOT NULL DEFAULT (datetime('now')),
    last_heartbeat   TEXT,
    last_claim_id    INTEGER
);
"""


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    # WAL keeps autopilot's writes from blocking MCP-tool reads on the same DB.
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.DatabaseError:
        pass
    return conn


def ensure_schema(db_path: str | Path) -> None:
    """Create the claims + autopilot_state tables if missing. Idempotent."""
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)


# ---------------------------------------------------------------------------
# Dataclass + helpers
# ---------------------------------------------------------------------------

@dataclass
class Claim:
    id: int
    title: str
    payload: dict[str, Any]
    priority: int
    status: str
    claimed_by: str | None
    stealable_at: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None
    error: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Claim":
        try:
            payload = json.loads(row["payload"] or "{}")
        except json.JSONDecodeError:
            payload = {}
        return cls(
            id=row["id"],
            title=row["title"],
            payload=payload,
            priority=row["priority"],
            status=row["status"],
            claimed_by=row["claimed_by"],
            stealable_at=row["stealable_at"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            error=row["error"],
        )


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def insert_claim(
    db_path: str | Path,
    *,
    title: str,
    payload: dict[str, Any] | None = None,
    priority: int = 5,
    stealable_at: str | datetime | None = None,
) -> int:
    """Insert a pending claim. Returns the new claim's id."""
    ensure_schema(db_path)
    if isinstance(stealable_at, datetime):
        if stealable_at.tzinfo is None:
            stealable_at = stealable_at.replace(tzinfo=timezone.utc)
        stealable_at = stealable_at.astimezone(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO claims(title, payload, priority, stealable_at)"
            " VALUES(?,?,?,?)",
            (title, json.dumps(payload or {}), int(priority), stealable_at),
        )
        conn.commit()
        return int(cur.lastrowid or 0)


def list_claims(
    db_path: str | Path,
    *,
    status: str | None = None,
    limit: int = 100,
) -> list[Claim]:
    """Return claims (newest first), optionally filtered by status."""
    ensure_schema(db_path)
    sql = "SELECT * FROM claims"
    args: list[Any] = []
    if status:
        sql += " WHERE status = ?"
        args.append(status)
    sql += " ORDER BY created_at DESC LIMIT ?"
    args.append(int(limit))
    with _connect(db_path) as conn:
        rows = conn.execute(sql, args).fetchall()
    return [Claim.from_row(r) for r in rows]


def claim_next(db_path: str | Path, runner_id: str) -> Claim | None:
    """Atomically pick the highest-priority stealable claim and reserve it.

    Returns ``None`` when the queue is empty (or every pending claim has
    ``stealable_at`` still in the future).
    """
    ensure_schema(db_path)
    with _connect(db_path) as conn:
        # `RETURNING *` lets us do select-and-update in a single statement.
        # Equivalent to background_jobs.dequeue_job in store.py.
        row = conn.execute(
            """
            UPDATE claims
            SET status     = 'running',
                claimed_by = ?,
                started_at = datetime('now')
            WHERE id = (
                SELECT id FROM claims
                WHERE status = 'pending'
                  AND claimed_by IS NULL
                  AND (stealable_at IS NULL OR stealable_at <= datetime('now'))
                ORDER BY priority ASC, created_at ASC
                LIMIT 1
            )
            RETURNING *
            """,
            (runner_id,),
        ).fetchone()
        if row is None:
            return None
        conn.commit()
        return Claim.from_row(row)


def mark_done(db_path: str | Path, claim_id: int) -> None:
    """Mark a claim as completed."""
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE claims SET status='done', finished_at=datetime('now'),"
            " error=NULL WHERE id=?",
            (claim_id,),
        )
        conn.commit()


def mark_failed(db_path: str | Path, claim_id: int, error: str) -> None:
    """Mark a claim as failed, with a short error string."""
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE claims SET status='failed', finished_at=datetime('now'),"
            " error=? WHERE id=?",
            (error[:1000], claim_id),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Runner state (used by autopilot_stop to signal a running loop)
# ---------------------------------------------------------------------------

def upsert_runner(db_path: str | Path, runner_id: str) -> None:
    """Insert the runner's row if missing; refresh ``started_at`` on conflict.

    NOTE: ``stop_requested`` is preserved across upserts so a previously-set
    stop flag survives a runner restart with the same id (the test for the
    SQLite stop signal depends on this). The runner is responsible for
    clearing the flag explicitly via :func:`clear_stop_requested` if it
    intends to reuse a stopped runner_id.
    """
    ensure_schema(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO autopilot_state(runner_id, stop_requested, started_at)"
            " VALUES(?, 0, datetime('now'))"
            " ON CONFLICT(runner_id) DO UPDATE SET"
            "   started_at     = datetime('now'),"
            "   last_heartbeat = datetime('now')",
            (runner_id,),
        )
        conn.commit()


def clear_stop_requested(db_path: str | Path, runner_id: str) -> None:
    """Explicitly clear the stop flag for a runner before reuse."""
    ensure_schema(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE autopilot_state SET stop_requested=0 WHERE runner_id=?",
            (runner_id,),
        )
        conn.commit()


def heartbeat(db_path: str | Path, runner_id: str, last_claim_id: int | None = None) -> None:
    """Update the runner's heartbeat timestamp."""
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE autopilot_state SET last_heartbeat=datetime('now'),"
            " last_claim_id=COALESCE(?, last_claim_id) WHERE runner_id=?",
            (last_claim_id, runner_id),
        )
        conn.commit()


def request_runner_stop(db_path: str | Path, runner_id: str = "") -> int:
    """Flip ``stop_requested`` for the named runner (or all runners if empty).

    Returns the number of rows touched.
    """
    ensure_schema(db_path)
    with _connect(db_path) as conn:
        if runner_id:
            cur = conn.execute(
                "UPDATE autopilot_state SET stop_requested=1 WHERE runner_id=?",
                (runner_id,),
            )
        else:
            cur = conn.execute("UPDATE autopilot_state SET stop_requested=1")
        conn.commit()
        return cur.rowcount or 0


def is_stop_requested(db_path: str | Path, runner_id: str) -> bool:
    """Read ``stop_requested`` for a runner. Missing rows are treated as 0."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT stop_requested FROM autopilot_state WHERE runner_id=?",
            (runner_id,),
        ).fetchone()
    return bool(row[0]) if row else False
