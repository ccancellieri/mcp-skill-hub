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
    # Disabled by default — enable explicitly via the cron UI or config.
    ("log-digest-snapshot",     "0 6 * * *",   "log_digest_snapshot",         0),
    ("wiki-reindex-nightly",    "0 5 * * *",   "wiki_reindex_nightly",        0),
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
    "0 5 * * *":    "daily at 5:00 AM",
    "0 6 * * *":    "daily at 6:00 AM",
    "0 0 * * 0":    "weekly on Sunday at midnight",
    "*/15 * * * *": "every 15 minutes",
}


def _wiki_reindex_nightly_handler() -> None:
    """Cron handler: rebuild wiki pages/edges and re-embed into vector namespaces.

    Disabled by default (enabled=0 in _DEFAULT_JOBS). Enable via the cron UI
    once the wiki vault has content worth nightly re-indexing.
    """
    from .store import get_store
    from . import wiki as _wiki
    from . import config as _cfg
    from pathlib import Path
    store = get_store()
    wiki_root = Path(_cfg.get("wiki_root") or
                     Path.home() / ".claude" / "mcp-skill-hub" / "wiki")
    result = _wiki.reindex(store, wiki_root, dry_run=False)
    _log.info(
        "wiki_reindex_nightly: pages=%d edges=%d vectors=%d",
        result["pages"], result["edges"], result["vectors"],
    )


def _log_digest_snapshot_handler() -> None:
    """Cron handler: build a log digest and write it to the standard logger.

    Disabled by default (enabled=0 in _DEFAULT_JOBS). Enable via the cron UI
    or set the cron_jobs table row to enabled=1. No side effects beyond reading
    the DB and emitting a log line — safe to run unattended once enabled.
    """
    from .log_insights import build_digest
    import json as _json
    d = build_digest(hours=24)
    _log.info(
        "log_digest_snapshot: total=%d sessions=%d failures=%d skills=%d",
        d["total"], d["distinct_sessions"],
        len(d["top_failures"]), len(d["skills"]),
    )
    _log.debug("log_digest_snapshot: %s", _json.dumps(d, default=str))


def _feedback_to_teachings_handler() -> None:
    """Cron handler: scan feedback_*.md memory files and report how many are present.

    The full "promote feedback rules into the teachings table" feature is not yet
    implemented (``continuous_teaching_enabled`` is False by default and no
    conversion function exists).  This handler is a safe, read-only probe that
    logs the count of feedback files and the number of teachings already in the
    DB — giving operators visibility without modifying any state.  When the full
    conversion logic is built it should replace the body of this function.
    """
    from pathlib import Path
    from .store import get_store

    memory_root = Path.home() / ".claude" / "projects"
    feedback_files: list[Path] = []
    if memory_root.exists():
        feedback_files = sorted(memory_root.rglob("feedback_*.md"))

    store = get_store()
    teaching_count = store.count_teachings()

    _log.info(
        "feedback_to_teachings: feedback_files=%d existing_teachings=%d"
        " (conversion not yet implemented — read-only probe)",
        len(feedback_files),
        teaching_count,
    )


def _optimize_memory_dry_run_handler() -> None:
    """Cron handler: run memory optimisation in dry-run mode (report only, no writes).

    Delegates to ``server.optimize_memory(dry_run=True)``.  The function has an
    internal IDLE-pressure gate — if the machine is not idle the call returns a
    skip message (no exception), so the cron record will still show ``ok``.
    If the LLM capability is not configured the scheduler records the error and
    moves on; nothing is corrupted.
    """
    from . import server as _server
    result = _server.optimize_memory(dry_run=True)
    _log.info("optimize_memory_dry_run: %s", str(result)[:200])


def _archive_memory_to_db_dry_run_handler() -> None:
    """Cron handler: dry-run pass over closed event sessions (no rows deleted).

    Calls ``store.events_prune(dry_run=True)`` which counts candidates and
    rows that *would* be coalesced without touching the database.
    """
    from .store import get_store
    store = get_store()
    result = store.events_prune(dry_run=True)
    _log.info(
        "archive_memory_to_db_dry_run: candidates=%d would_delete=%d",
        result["candidates"],
        result["rows_deleted"],
    )


def _memexp_snapshot_create_handler() -> None:
    """Cron handler: write a weekly training-data snapshot to disk.

    Calls ``store.export_training_data()`` and writes three JSONL files
    (feedback, triage, compact) to ``~/.claude/mcp-skill-hub/training/``.
    The output directory is created if absent; existing files are overwritten.
    Safe to run unattended — read-only against the DB, writes only to the
    snapshot directory.
    """
    import json as _json
    from pathlib import Path
    from .store import get_store

    store = get_store()
    data = store.export_training_data()

    out_dir = Path.home() / ".claude" / "mcp-skill-hub" / "training"
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}

    fb_path = out_dir / "feedback.jsonl"
    with fb_path.open("w", encoding="utf-8") as fh:
        for pair in data["feedback_pairs"]:
            fh.write(_json.dumps({
                "instruction": (
                    "Given a user query, rate whether this skill is relevant (true/false)."
                ),
                "input": (
                    f"Query: {pair['query']}\n"
                    f"Skill: {pair['skill_description']}\n"
                    f"Content: {pair['skill_content'][:500]}"
                ),
                "output": str(pair["label"]).lower(),
                "metadata": {"skill_id": pair["skill_id"]},
            }) + "\n")
    counts["feedback"] = len(data["feedback_pairs"])

    triage_path = out_dir / "triage.jsonl"
    with triage_path.open("w", encoding="utf-8") as fh:
        for pair in data["triage_pairs"]:
            fh.write(_json.dumps({
                "instruction": (
                    "Classify this user message for a coding assistant. "
                    "Actions: local_answer, local_action, enrich_and_forward, pass_through."
                ),
                "input": pair["message"],
                "output": pair["action"],
                "metadata": {"confidence": pair["confidence"]},
            }) + "\n")
    counts["triage"] = len(data["triage_pairs"])

    compact_path = out_dir / "compact.jsonl"
    with compact_path.open("w", encoding="utf-8") as fh:
        for pair in data["compact_pairs"]:
            fh.write(_json.dumps({
                "title": pair["title"],
                "input": pair["input"],
                "output": pair["output"],
            }) + "\n")
    counts["compact"] = len(data["compact_pairs"])

    _log.info(
        "memexp_snapshot_create: feedback=%d triage=%d compact=%d -> %s",
        counts["feedback"], counts["triage"], counts["compact"], out_dir,
    )


def _check_embedding_backends_handler() -> None:
    """Cron handler: probe the configured embedding backend and log reachability.

    Calls ``embeddings.embed_available()`` (no network I/O — checks config/env
    only) and logs the result.  Runs every 15 minutes so pipeline health is
    visible in the log stream without a browser.
    """
    from . import embeddings as _emb
    from . import config as _cfg

    available = _emb.embed_available()
    priority = _cfg.get("embedding_backend_priority") or [
        "voyage", "ollama", "sentence_transformers"
    ]
    backend = priority[0] if isinstance(priority, list) and priority else "unknown"
    if available:
        _log.info(
            "check_embedding_backends: backend=%s reachable=True", backend
        )
    else:
        reason = _emb.embed_unavailable_reason()
        _log.warning(
            "check_embedding_backends: backend=%s reachable=False reason=%s",
            backend, reason,
        )


# Register module-level handlers so the scheduler can dispatch them.
_HANDLERS["log_digest_snapshot"] = _log_digest_snapshot_handler
_HANDLERS["wiki_reindex_nightly"] = _wiki_reindex_nightly_handler
_HANDLERS["optimize_memory_dry_run"] = _optimize_memory_dry_run_handler
_HANDLERS["feedback_to_teachings"] = _feedback_to_teachings_handler
_HANDLERS["archive_memory_to_db_dry_run"] = _archive_memory_to_db_dry_run_handler
_HANDLERS["memexp_snapshot_create"] = _memexp_snapshot_create_handler
_HANDLERS["check_embedding_backends"] = _check_embedding_backends_handler


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
        self._running_jobs: set[str] = set()
        self._running_lock = threading.Lock()

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
        with self._running_lock:
            if name in self._running_jobs:
                _log.debug("cron: job %r already running, skipping", name)
                return
            self._running_jobs.add(name)
        try:
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
        finally:
            with self._running_lock:
                self._running_jobs.discard(name)


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
