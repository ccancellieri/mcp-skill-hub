"""Simple cron scheduler backed by the SQLite cron_jobs table."""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Handler registry — command name → Python callable
# ---------------------------------------------------------------------------

_HANDLERS: dict[str, Callable[[], None]] = {}


def register_handler(name: str, fn: Callable[[], None]) -> None:
    """Register a callable for the given command name."""
    _HANDLERS[name] = fn


# ---------------------------------------------------------------------------
# Default job definitions — seeded when table is empty
# ---------------------------------------------------------------------------

_DEFAULT_JOBS = [
    ("memory-optimize-preview", "0 2 * * *",   "optimize_memory_dry_run",    1),
    ("teachings-sync",          "0 3 * * *",   "feedback_to_teachings",       1),
    ("archive-closed-tasks",    "0 4 * * *",   "archive_memory_to_db_dry_run", 1),
    ("memory-export-snapshot",  "0 0 * * 0",   "memexp_snapshot_create",      0),
    ("pipeline-health-check",   "*/15 * * * *", "check_embedding_backends",   1),
]


def seed_defaults(db_path: str) -> None:
    """Insert default cron jobs if the cron_jobs table is empty."""
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT count(*) FROM cron_jobs").fetchone()[0]
        if count == 0:
            conn.executemany(
                "INSERT OR IGNORE INTO cron_jobs(name, schedule, command, enabled)"
                " VALUES(?,?,?,?)",
                _DEFAULT_JOBS,
            )


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------

_HUMAN_SCHEDULES: dict[str, str] = {
    "0 2 * * *":    "daily at 2:00 AM",
    "0 3 * * *":    "daily at 3:00 AM",
    "0 4 * * *":    "daily at 4:00 AM",
    "0 0 * * 0":    "weekly on Sunday at midnight",
    "*/15 * * * *": "every 15 minutes",
}


def human_schedule(schedule: str) -> str:
    """Return a human-readable description of a cron expression."""
    return _HUMAN_SCHEDULES.get(schedule, schedule)


def next_run_from(schedule: str, after: datetime | None = None) -> datetime | None:
    """Return the next scheduled datetime for a cron expression."""
    try:
        from croniter import croniter  # type: ignore
        base = after or datetime.now(timezone.utc)
        # croniter wants a naive or tz-aware datetime; pass utcoffset-aware.
        return croniter(schedule, base).get_next(datetime)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class CronScheduler:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="cron-scheduler"
        )
        self._thread.start()
        _log.info("CronScheduler started (db=%s)", self._db_path)

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(timeout=60):
            self._tick()

    def _tick(self) -> None:
        """Check for due jobs and execute them."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                jobs = conn.execute(
                    "SELECT * FROM cron_jobs WHERE enabled=1"
                ).fetchall()
            for row in jobs:
                self._maybe_run(dict(row))
        except Exception as exc:
            _log.warning("cron tick error: %s", exc)

    def _maybe_run(self, job: dict) -> None:
        last = job.get("last_run_at")
        if last:
            try:
                last_dt = datetime.fromisoformat(last)
                # Normalise to UTC-aware if naive (SQLite stores as UTC text)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
            except ValueError:
                last_dt = datetime(2000, 1, 1, tzinfo=timezone.utc)
        else:
            last_dt = datetime(2000, 1, 1, tzinfo=timezone.utc)

        nxt = next_run_from(job["schedule"], last_dt)
        if nxt is None:
            return
        # Ensure nxt is tz-aware for comparison
        if nxt.tzinfo is None:
            nxt = nxt.replace(tzinfo=timezone.utc)
        if nxt > datetime.now(timezone.utc):
            return
        self._run_job(job)

    def _run_job(self, job: dict) -> None:
        name = job["name"]
        command = job["command"]
        _log.info("cron: running job %r (command=%r)", name, command)
        handler = _HANDLERS.get(command)
        if handler is None:
            _log.debug("cron: no handler for %r, skipping", command)
            return
        start = time.monotonic()
        status = "ok"
        error: str | None = None
        try:
            handler()
        except Exception as exc:
            status = "error"
            error = str(exc)[:500]
            _log.warning("cron: job %r failed: %s", name, exc)
        duration_ms = int((time.monotonic() - start) * 1000)
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "UPDATE cron_jobs"
                    " SET last_run_at=datetime('now'), last_status=?,"
                    " last_error=?, last_duration_ms=?, run_count=run_count+1"
                    " WHERE name=?",
                    (status, error, duration_ms, name),
                )
        except Exception as exc:
            _log.warning("cron: failed to update job state for %r: %s", name, exc)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_SCHEDULER: CronScheduler | None = None


def get_scheduler(db_path: str | None = None) -> CronScheduler:
    global _SCHEDULER
    if _SCHEDULER is None:
        if db_path is None:
            from .store import DB_PATH
            db_path = str(DB_PATH)
        _SCHEDULER = CronScheduler(db_path)
    return _SCHEDULER
