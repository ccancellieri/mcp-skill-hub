"""drop_dead_ddl.py — offline migration dropping schema left over after the
#130 dead-code purge.

The model-reward bandit (#120) and the task-claims layer were removed from
the codebase in #130, but their schema stayed behind because dropping
columns/tables is a migration, not something the daemon should do at
startup. This script removes it:

- DROP TABLE model_rewards
- DROP INDEX idx_tasks_claimed_by
- ALTER TABLE tasks DROP COLUMN claimed_by / claim_token / claimed_at / stealable_at

Safety: refuses to run unless it can immediately acquire a write lock on the
database (a busy/locked DB means another process — e.g. the running daemon —
is using it). Never run this against a DB that a live skill-hub process has
open; stop the daemon first.

Usage:
    uv run python scripts/drop_dead_ddl.py [DB_PATH] [--vacuum]

Default DB_PATH: ~/.claude/mcp-skill-hub/skill_hub.db
--vacuum: rewrite the whole file after dropping columns (slow on large DBs;
          reclaims the space they occupied instead of leaving it as free
          pages).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

_DEFAULT_DB_PATH = Path.home() / ".claude" / "mcp-skill-hub" / "skill_hub.db"

_CLAIM_COLUMNS = ("claimed_by", "claim_token", "claimed_at", "stealable_at")


def _acquire_exclusive_write_lock(conn: sqlite3.Connection, db_path: Path) -> None:
    """Refuse to proceed unless the DB is not being written by anyone else.

    Uses a zero busy-timeout ``BEGIN IMMEDIATE``: if another connection (e.g.
    the live daemon) currently holds the write lock, this fails right away
    with ``sqlite3.OperationalError`` instead of blocking. The transaction is
    left open on success — the caller runs the migration inside it.
    """
    conn.execute("PRAGMA busy_timeout = 0")
    try:
        conn.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError as exc:
        raise SystemExit(
            f"refusing to run: could not acquire an exclusive write lock on "
            f"{db_path} ({exc}). Stop the skill-hub daemon before running "
            "this migration."
        ) from exc

    wal_path = db_path.with_name(db_path.name + "-wal")
    if wal_path.exists() and wal_path.stat().st_size > 0:
        print(
            f"note: {wal_path} has {wal_path.stat().st_size} bytes of "
            "uncheckpointed WAL activity; the exclusive lock succeeded so "
            "proceeding, but confirm no other process is using this DB.",
            file=sys.stderr,
        )


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _index_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    return column in cols


def run(db_path: Path, *, vacuum: bool = False) -> None:
    if not db_path.exists():
        raise SystemExit(f"no such database: {db_path}")

    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        _acquire_exclusive_write_lock(conn, db_path)

        try:
            if _table_exists(conn, "model_rewards"):
                conn.execute("DROP TABLE IF EXISTS model_rewards")
                print("dropped table model_rewards")
            else:
                print("model_rewards already absent, skipping")

            if _index_exists(conn, "idx_tasks_claimed_by"):
                conn.execute("DROP INDEX IF EXISTS idx_tasks_claimed_by")
                print("dropped index idx_tasks_claimed_by")
            else:
                print("idx_tasks_claimed_by already absent, skipping")

            if _table_exists(conn, "tasks"):
                for col in _CLAIM_COLUMNS:
                    if _column_exists(conn, "tasks", col):
                        conn.execute(f"ALTER TABLE tasks DROP COLUMN {col}")
                        print(f"dropped column tasks.{col}")
                    else:
                        print(f"tasks.{col} already absent, skipping")

            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        if vacuum:
            conn.execute("VACUUM")
    finally:
        conn.close()

    print(f"done: {db_path}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Offline migration: drop the model_rewards table and the "
        "task claim columns left over after the #130 dead-code purge (#131)."
    )
    p.add_argument(
        "db_path",
        nargs="?",
        type=Path,
        default=_DEFAULT_DB_PATH,
        help=f"Path to the skill-hub SQLite DB (default: {_DEFAULT_DB_PATH}).",
    )
    p.add_argument(
        "--vacuum",
        action="store_true",
        default=False,
        help="Run VACUUM after dropping the dead schema (rewrites the whole "
        "file; slow on large DBs).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    run(args.db_path, vacuum=args.vacuum)


if __name__ == "__main__":
    main()
