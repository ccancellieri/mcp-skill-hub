"""Background job dispatcher for the skill hub.

This module exposes two layers:

1. **Queue layer** (module-level functions) — manages job state in SQLite,
   builds housekeeping blocks for the session_start_enforcer hook.  These
   functions use a raw db_path and keep their own connections so they can be
   called from hook processes that do not own a SkillStore.

2. **Dispatcher layer** (``JobDispatcher`` class) — wraps a ``SkillStore``
   and runs jobs synchronously or in a background thread.  Priority order
   (configurable via ``background_worker_priority`` config):

       1. litellm  — cloud LLM via litellm (uses API key if set)
       2. ollama   — local/remote Ollama endpoint
       3. defer    — re-queue, retry on next tick

   The ``subagent`` worker is handled at the hook layer
   (session_start_enforcer.py) by injecting a housekeeping block; it is
   listed in the default config priority but is not dispatched here.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Callable

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema bootstrap (used by tests; production uses SkillStore._migrate)
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS background_jobs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    kind         TEXT NOT NULL,
    payload      TEXT NOT NULL DEFAULT '{}',
    priority     INTEGER NOT NULL DEFAULT 5,
    status       TEXT NOT NULL DEFAULT 'pending'
                     CHECK(status IN ('pending','running','done','failed','deferred')),
    worker_used  TEXT,
    result       TEXT,
    error        TEXT,
    attempts     INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    started_at   TEXT,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_background_jobs_status_priority
    ON background_jobs(status, priority, created_at);
"""


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Idempotent schema bootstrap — used when the caller owns the connection."""
    conn.executescript(_SCHEMA_SQL)
    conn.commit()


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class BackgroundJob:
    id: int
    kind: str
    payload: dict
    priority: int
    status: str
    attempts: int
    worker_used: str | None = None
    result: dict | None = None
    error: str | None = None
    created_at: str = ""


def _row_to_job(row: sqlite3.Row) -> BackgroundJob:
    d = dict(row)
    payload: dict = {}
    try:
        payload = json.loads(d.get("payload") or "{}")
    except (json.JSONDecodeError, TypeError):
        pass
    result: dict | None = None
    raw_result = d.get("result")
    if raw_result:
        try:
            result = json.loads(raw_result)
        except (json.JSONDecodeError, TypeError):
            result = {"raw": raw_result}
    return BackgroundJob(
        id=d["id"],
        kind=d["kind"],
        payload=payload,
        priority=d["priority"],
        status=d["status"],
        attempts=d["attempts"],
        worker_used=d.get("worker_used"),
        result=result,
        error=d.get("error"),
        created_at=d.get("created_at") or "",
    )


# ---------------------------------------------------------------------------
# Queue API — module-level functions (used by session_start_enforcer hook)
# ---------------------------------------------------------------------------


def enqueue_job(
    db_path: str,
    kind: str,
    payload: dict,
    priority: int = 5,
) -> int:
    """Insert a new pending job. Returns the job id."""
    conn = _connect(db_path)
    try:
        _ensure_schema(conn)
        cur = conn.execute(
            """
            INSERT INTO background_jobs (kind, payload, priority, status)
            VALUES (?, ?, ?, 'pending')
            """,
            (kind, json.dumps(payload), priority),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]
    finally:
        conn.close()


def list_pending_jobs(db_path: str, max_jobs: int = 3) -> list[BackgroundJob]:
    """Return up to *max_jobs* pending jobs ordered by priority ASC, created_at ASC."""
    conn = _connect(db_path)
    try:
        _ensure_schema(conn)
        rows = conn.execute(
            """
            SELECT * FROM background_jobs
            WHERE status = 'pending'
            ORDER BY priority ASC, created_at ASC
            LIMIT ?
            """,
            (max_jobs,),
        ).fetchall()
        return [_row_to_job(r) for r in rows]
    finally:
        conn.close()


def mark_running(db_path: str, job_id: int) -> None:
    """Set status=running, started_at=now, attempts+=1."""
    conn = _connect(db_path)
    try:
        _ensure_schema(conn)
        conn.execute(
            """
            UPDATE background_jobs
            SET status = 'running',
                started_at = datetime('now'),
                attempts = attempts + 1
            WHERE id = ?
            """,
            (job_id,),
        )
        conn.commit()
    finally:
        conn.close()


def mark_done(db_path: str, job_id: int, result: dict, worker_used: str) -> None:
    """Set status=done, result=json(result), worker_used, completed_at=now."""
    conn = _connect(db_path)
    try:
        _ensure_schema(conn)
        conn.execute(
            """
            UPDATE background_jobs
            SET status = 'done',
                result = ?,
                worker_used = ?,
                completed_at = datetime('now')
            WHERE id = ?
            """,
            (json.dumps(result), worker_used, job_id),
        )
        conn.commit()
    finally:
        conn.close()


def mark_failed(db_path: str, job_id: int, error: str) -> None:
    """Increment attempts. If attempts >= retry_max → status=failed, else status=pending.

    The retry_max is read from config; defaults to 3 if config is unavailable.
    """
    retry_max = 3
    try:
        from skill_hub import config as _cfg
        retry_max = int(_cfg.get("background_job_retry_max") or 3)
    except Exception:
        pass

    conn = _connect(db_path)
    try:
        _ensure_schema(conn)
        row = conn.execute(
            "SELECT attempts FROM background_jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if row is None:
            return
        new_attempts = (row["attempts"] or 0) + 1
        new_status = "failed" if new_attempts >= retry_max else "pending"
        conn.execute(
            """
            UPDATE background_jobs
            SET status = ?,
                error = ?,
                attempts = ?,
                completed_at = CASE WHEN ? = 'failed' THEN datetime('now') ELSE NULL END
            WHERE id = ?
            """,
            (new_status, error, new_attempts, new_status, job_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_pending_count(db_path: str) -> int:
    """Return count of pending+deferred jobs."""
    conn = _connect(db_path)
    try:
        _ensure_schema(conn)
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM background_jobs WHERE status IN ('pending','deferred')"
        ).fetchone()
        return int(row["cnt"]) if row else 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Dispatch helpers (used by session_start_enforcer hook)
# ---------------------------------------------------------------------------


def should_dispatch(
    last_user_message_ts: float,
    idle_threshold_ms: int = 3000,
) -> bool:
    """Return True if the user has been idle long enough to run a background job."""
    return (time.time() - last_user_message_ts) * 1000 >= idle_threshold_ms


def build_housekeeping_block(jobs: list[BackgroundJob]) -> str:
    """Build a system message block asking Claude to handle one background job.

    Only the first job is handled per invocation.

    Returns a string like::

        ---HOUSEKEEPING---
        Background job pending (kind=classify_file, id=42):
        {"path": "/some/file.md"}
        Please handle this by spawning: Agent(description="housekeeping: classify_file",
        subagent_type="general", prompt="Classify this memory file for archival: ...")
        ---END HOUSEKEEPING---
    """
    if not jobs:
        return ""
    job = jobs[0]
    payload_str = json.dumps(job.payload, indent=2)
    return (
        "---HOUSEKEEPING---\n"
        f"Background job pending (kind={job.kind}, id={job.id}):\n"
        f"{payload_str}\n"
        f'Please handle this by spawning: Agent(description="housekeeping: {job.kind}",\n'
        f'subagent_type="general", prompt="Handle background job kind={job.kind} id={job.id}: '
        f"{payload_str.replace(chr(10), ' ')}\")\n"
        "---END HOUSEKEEPING---"
    )


# ---------------------------------------------------------------------------
# Handler registry — JobDispatcher worker layer
# ---------------------------------------------------------------------------

# Registry of job kind → handler function.
# Signature: handler(payload: dict) -> dict (result)
_HANDLERS: dict[str, Callable[[dict], dict]] = {}


def register_handler(kind: str) -> Callable:
    """Decorator: register a function as the handler for a job kind."""
    def decorator(fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
        _HANDLERS[kind] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# JobDispatcher
# ---------------------------------------------------------------------------


class JobDispatcher:
    """Dispatches pending background jobs using the configured worker priority."""

    def __init__(self, store=None) -> None:
        from . import config as _cfg
        from .store import SkillStore
        self._store: SkillStore = store or SkillStore()
        self._cfg = _cfg
        self._lock = threading.Lock()
        self._running = False

    # ------------------------------------------------------------------
    # Public API

    def enqueue(self, kind: str, payload: dict | None = None, priority: int = 5) -> int:
        """Enqueue a job and return its id."""
        return self._store.enqueue_job(kind, payload or {}, priority)

    def dispatch_one(self) -> bool:
        """Claim and run the next pending job. Returns True if a job was processed."""
        with self._lock:
            if self._running:
                return False
            row = self._store.dequeue_job()
            if row is None:
                return False
            self._running = True

        job_id = row["id"]
        kind = row["kind"]
        payload = json.loads(row["payload"] or "{}")

        try:
            result = self._run_with_worker(kind, payload)
            self._store.complete_job(job_id, result=result, worker="local")
            _log.debug("background_job %d (%s) completed", job_id, kind)
        except Exception as exc:
            _log.warning("background_job %d (%s) failed: %s", job_id, kind, exc)
            self._store.fail_job(job_id, str(exc))
        finally:
            with self._lock:
                self._running = False

        return True

    def dispatch_in_background(self) -> None:
        """Dispatch one job in a daemon thread (non-blocking)."""
        t = threading.Thread(target=self.dispatch_one, daemon=True)
        t.start()

    def pending_count(self) -> int:
        return self._store.pending_job_count()

    def reset_deferred(self) -> int:
        return self._store.reset_deferred_jobs()

    # ------------------------------------------------------------------
    # Internal

    def _run_with_worker(self, kind: str, payload: dict) -> dict:
        """Try the configured worker priority until one succeeds."""
        from . import config as _cfg
        priority_order: list[str] = list(
            _cfg.get("background_worker_priority") or ["litellm", "ollama", "defer"]
        )

        handler = _HANDLERS.get(kind)
        if handler is None:
            raise ValueError(f"no handler registered for job kind {kind!r}")

        errors: list[str] = []
        for worker in priority_order:
            if worker == "defer":
                raise RuntimeError(
                    f"all workers exhausted; deferred. errors: {'; '.join(errors)}"
                )
            if worker == "subagent":
                # subagent dispatch is handled at the hook layer; skip here
                continue
            try:
                return self._run_via_worker(worker, kind, payload, handler)
            except Exception as exc:
                errors.append(f"{worker}: {exc}")
                continue

        raise RuntimeError(f"no workers available: {'; '.join(errors)}")

    def _run_via_worker(
        self,
        worker: str,
        kind: str,
        payload: dict,
        handler: Callable[[dict], dict],
    ) -> dict:
        if worker == "litellm":
            return self._run_litellm(kind, payload, handler)
        if worker == "ollama":
            return self._run_ollama(kind, payload, handler)
        raise ValueError(f"unknown worker: {worker!r}")

    def _run_litellm(self, kind: str, payload: dict, handler: Callable) -> dict:
        """Run via litellm (Anthropic/cloud). Falls back if no API key."""
        from . import config as _cfg
        providers = _cfg.get("llm_providers") or {}
        mid_model = (providers or {}).get("tier_mid")
        if not mid_model:
            raise RuntimeError("no tier_mid model configured for litellm worker")
        # Invoke the handler directly — the handler itself uses get_provider()
        # which routes through litellm. The 'worker' abstraction here is about
        # which cloud provider's API key is available, not the execution method.
        return handler(payload)

    def _run_ollama(self, kind: str, payload: dict, handler: Callable) -> dict:
        """Run via Ollama (local/remote). Falls back if no healthy endpoint."""
        from .ollama_client import get_ollama_client
        client = get_ollama_client()
        if client.get_api_base(None) is None:
            raise RuntimeError("no healthy Ollama endpoint available")
        return handler(payload)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_DISPATCHER: JobDispatcher | None = None
_DISPATCHER_LOCK = threading.Lock()


def get_dispatcher(store=None) -> JobDispatcher:
    global _DISPATCHER
    with _DISPATCHER_LOCK:
        if _DISPATCHER is None:
            _DISPATCHER = JobDispatcher(store=store)
    return _DISPATCHER


# ---------------------------------------------------------------------------
# Built-in job handlers
# ---------------------------------------------------------------------------


@register_handler("noop")
def _handle_noop(payload: dict) -> dict:
    """No-op job for testing and health checks."""
    return {"ok": True, "payload_echo": payload}


@register_handler("optimize_memory_file")
def _handle_optimize_memory_file(payload: dict) -> dict:
    """Compact a single memory file using the configured LLM."""
    file_path = payload.get("file_path")
    if not file_path:
        raise ValueError("optimize_memory_file requires 'file_path' in payload")
    from pathlib import Path
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"memory file not found: {file_path}")
    content = path.read_text(encoding="utf-8")
    if len(content) < 200:
        return {"skipped": True, "reason": "file too short to compact"}
    from .embeddings import compact
    digest = compact(content[:4000])
    return {"compacted": True, "original_chars": len(content), "digest": digest}
