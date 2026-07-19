"""Simple cron scheduler backed by the SQLite cron_jobs table."""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
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
    ("memory-optimize", "0 2 * * *",   "optimize_memory_apply",    1),
    ("teachings-sync",          "0 3 * * *",   "feedback_to_teachings",       1),
    ("archive-closed-tasks",    "0 4 * * *",   "archive_memory_to_db_dry_run", 1),
    ("pipeline-health-check",   "*/15 * * * *", "check_embedding_backends",   1),
    # Keeps configured CodeGraph indexes fresh; inert until roots are set.
    ("codegraph-sync",          "*/30 * * * *", "codegraph_sync",             1),
    # Re-enables providers marked auto_reenable once /models answers again;
    # inert unless a provider actually carries the flag.
    ("provider-reprobe",        "*/15 * * * *", "reprobe_disabled_providers", 1),
    # Disabled by default — enable explicitly via the cron UI or config.
    ("log-digest-snapshot",        "0 6 * * *",   "log_digest_snapshot",         0),
    # L1/L2 plugin curation — decides which dormant plugins to disable. Only
    # writes settings.json when plugin_curation_auto is also set. Off by default.
    ("plugin-curation",            "0 5 * * *",   "plugin_curation",             0),
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
    "0 1 * * *":    "daily at 1:00 AM",
    "0 2 * * *":    "daily at 2:00 AM",
    "0 3 * * *":    "daily at 3:00 AM",
    "0 4 * * *":    "daily at 4:00 AM",
    "0 5 * * *":    "daily at 5:00 AM",
    "0 6 * * *":    "daily at 6:00 AM",
    "0 0 * * 0":    "weekly on Sunday at midnight",
    "*/15 * * * *": "every 15 minutes",
    "*/30 * * * *": "every 30 minutes",
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
    """Cron handler: promote feedback_*.md memory files into persistent teachings.

    Gated by ``continuous_teaching_enabled`` (False by default). While
    disabled, this stays the original read-only probe — it logs the count of
    feedback files and existing teachings without modifying any state, so the
    scheduled slot remains informative and harmless.

    Once enabled, delegates to :func:`feedback_teachings.convert`, which scans
    the same feedback source, de-duplicates against existing teaching rule
    text, and inserts new ones via ``store.add_teaching`` — the same
    embed()/insert flow the ``teach`` MCP tool uses.
    """
    from pathlib import Path
    from . import config as _cfg
    from .store import get_store

    memory_root = Path.home() / ".claude" / "projects"
    store = get_store()

    if not _cfg.get("continuous_teaching_enabled"):
        feedback_files: list[Path] = []
        if memory_root.exists():
            feedback_files = sorted(memory_root.rglob("feedback_*.md"))
        teaching_count = store.count_teachings()
        _log.info(
            "feedback_to_teachings: feedback_files=%d existing_teachings=%d"
            " (continuous_teaching_enabled=False — read-only probe)",
            len(feedback_files),
            teaching_count,
        )
        return

    from . import feedback_teachings

    result = feedback_teachings.convert(store, memory_root)
    _log.info(
        "feedback_to_teachings: found=%d converted=%d duplicates=%d"
        " unparsed=%d errors=%d",
        result["found"], result["converted"], result["duplicates"],
        result["unparsed"], result["errors"],
    )


def _plugin_curation_handler() -> None:
    """Cron handler: L1/L2 decide which dormant plugins to disable.

    Gated by ``plugin_curation_enabled``; only writes ``~/.claude/settings.json``
    when ``plugin_curation_auto`` is also set (otherwise it logs recommendations
    for a human/the main LLM to apply). Disabled by default in ``_DEFAULT_JOBS``.
    """
    from . import plugin_curation
    from .store import get_store

    results = plugin_curation.run_curation(get_store())
    if not results:
        _log.info("plugin_curation: no dormant-plugin decisions this pass")
        return
    for r in results:
        _log.info(
            "plugin_curation: %s -> %s%s (%s)",
            r["plugin_id"], r["action"],
            " [applied]" if r["applied"] else "",
            r["reason"],
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


# Where nightly memory snapshots are kept before an apply pass, and how many to
# retain (one month of nightly runs).
_MEM_BACKUP_DIR = Path.home() / ".claude" / "mcp-skill-hub" / "memory-backups"
_MEM_BACKUP_KEEP = 30


def _snapshot_memory_dir() -> Path | None:
    """Copy the auto-memory dir to a timestamped backup before a destructive pass.

    Returns the backup path, or None if the memory dir does not exist. Prunes
    old snapshots beyond ``_MEM_BACKUP_KEEP`` so backups never grow unbounded.
    Best-effort: a copy failure raises so the caller can abort the apply.
    """
    import shutil
    import time as _time

    mem_path = (Path.home() / ".claude" / "projects"
                / "-Users-ccancellieri-work-code" / "memory")
    if not mem_path.exists():
        return None
    _MEM_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    dest = _MEM_BACKUP_DIR / _time.strftime("%Y%m%d-%H%M%S")
    shutil.copytree(mem_path, dest, dirs_exist_ok=True)
    # Retention: keep the newest N snapshots, drop the rest.
    snaps = sorted(p for p in _MEM_BACKUP_DIR.iterdir() if p.is_dir())
    for old in snaps[:-_MEM_BACKUP_KEEP]:
        shutil.rmtree(old, ignore_errors=True)
    return dest


def _optimize_memory_apply_handler() -> None:
    """Cron handler: APPLY memory optimisation (deletes PRUNE-flagged files).

    Snapshots the memory dir first (reversible), then runs
    ``optimize_memory(dry_run=False)``. Two safety layers remain in force:
    the IDLE-pressure gate inside ``optimize_memory`` (skips under load), and
    the LLM escalation ladder — if the local Ollama daemon is down the call is
    automatically routed to the gateway/personal provider tier instead of
    failing, so the pass still runs when Ollama is unavailable.
    """
    from . import server as _server
    backup = _snapshot_memory_dir()
    result = _server.optimize_memory(dry_run=False)
    _log.info("optimize_memory_apply: backup=%s result=%s",
              backup, str(result)[:200])


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


def _discussions_sync_nightly_handler() -> None:
    """Cron handler: periodic full idempotent sync of GitHub Discussions into wiki.

    Disabled by default (enabled=0 in _DEFAULT_JOBS). Enable via the cron UI
    once a GitHub repo is configured and discussions_write_enabled is not
    required (this handler only runs the read-path sync_discussions).

    Each run performs a full idempotent sync (fetches up to 50 discussions,
    upserts them as wiki source pages). Because write_source_page is
    content-addressed (source_hash), unchanged discussions are skipped cheaply.
    """
    from .store import get_store
    from . import discussions_sync as _disc
    from . import config as _cfg

    store = get_store()
    repo = str(_cfg.get("discussions_repo") or "")
    result = _disc.sync_discussions(store, repo=repo, dry_run=False)
    if "error" in result:
        _log.warning(
            "discussions_sync_nightly error: %s", result["error"]
        )
    else:
        _log.info(
            "discussions_sync_nightly: checked=%d indexed=%d discussions=%d"
            " comments=%d skipped=%d",
            result["checked"], result["indexed"], result["discussions"],
            result["comments"], result["skipped"],
        )


def _codegraph_sync_handler() -> None:
    """Cron handler: keep configured CodeGraph indexes fresh (incremental sync).

    For each repo in ``codegraph_reindex_roots`` that already has a
    ``.codegraph/`` index, shells out to ``codegraph sync`` — which only
    processes files changed since the last index, so it is near-instant when
    nothing changed. Deterministic (no LLM), bounded (per-repo timeout), and a
    cheap no-op when the roots list is empty or the binary is absent.

    Enabled by default but inert until roots are configured, so a fresh install
    does no work while a configured one stays continuously up to date.
    """
    import subprocess
    from pathlib import Path
    from . import config as _cfg
    from .codegraph_context import has_codegraph_index, _find_codegraph_bin

    roots = _cfg.get("codegraph_reindex_roots") or []
    if not isinstance(roots, list) or not roots:
        return
    bin_path = _find_codegraph_bin()
    if bin_path is None:
        _log.debug("codegraph_sync: binary not found; skipping")
        return
    timeout = float(_cfg.get("codegraph_reindex_timeout_seconds") or 120)

    synced = 0
    skipped = 0
    for raw in roots:
        try:
            root = Path(str(raw)).expanduser()
            if not has_codegraph_index(root):
                skipped += 1
                continue
            result = subprocess.run(
                [bin_path, "sync", "--quiet", str(root)],
                capture_output=True, timeout=timeout, check=False,
            )
            if result.returncode == 0:
                synced += 1
            else:
                skipped += 1
                _log.warning(
                    "codegraph_sync: %s exited %d: %s", root, result.returncode,
                    result.stderr.decode(errors="replace")[:120],
                )
        except subprocess.TimeoutExpired:
            skipped += 1
            _log.warning("codegraph_sync: timed out for %s", raw)
        except Exception as exc:  # noqa: BLE001 — best-effort per repo
            skipped += 1
            _log.warning("codegraph_sync: failed for %s: %s", raw, exc)
    _log.info("codegraph_sync: synced=%d skipped=%d", synced, skipped)


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
        "ollama", "sentence_transformers"
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


def _reprobe_disabled_providers_handler() -> None:
    """Cron handler: re-enable transiently-disabled providers that recovered.

    Off unless ``provider_reprobe_enabled`` is set. Probes each disabled
    provider marked ``auto_reenable`` (see :mod:`skill_hub.llm.health`) and logs
    any that came back so the escalation ladder can use them again. Only ever
    re-enables — never disables — so it is safe to run unattended.
    """
    from . import config as _cfg

    if not _cfg.get("provider_reprobe_enabled"):
        return
    from .llm import health

    try:
        changed = health.reprobe_disabled_providers()
    except Exception as exc:  # a probe error must never kill the scheduler thread
        _log.warning("reprobe_disabled_providers: probe failed: %s", exc)
        return
    for c in changed:
        _log.info(
            "reprobe_disabled_providers: re-enabled provider=%s (status=%s)",
            c.get("name"), c.get("status"),
        )


# Register module-level handlers so the scheduler can dispatch them.
_HANDLERS["log_digest_snapshot"] = _log_digest_snapshot_handler
_HANDLERS["wiki_reindex_nightly"] = _wiki_reindex_nightly_handler
_HANDLERS["optimize_memory_dry_run"] = _optimize_memory_dry_run_handler
_HANDLERS["optimize_memory_apply"] = _optimize_memory_apply_handler
_HANDLERS["feedback_to_teachings"] = _feedback_to_teachings_handler
_HANDLERS["archive_memory_to_db_dry_run"] = _archive_memory_to_db_dry_run_handler
_HANDLERS["memexp_snapshot_create"] = _memexp_snapshot_create_handler
_HANDLERS["check_embedding_backends"] = _check_embedding_backends_handler
_HANDLERS["codegraph_sync"] = _codegraph_sync_handler
_HANDLERS["discussions_sync_nightly"] = _discussions_sync_nightly_handler
_HANDLERS["plugin_curation"] = _plugin_curation_handler
_HANDLERS["reprobe_disabled_providers"] = _reprobe_disabled_providers_handler


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
