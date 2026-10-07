"""Index freshness — periodic reindex sweep + task-close refresh (#134).

The wiki and vector indexes only updated when someone ran a reindex tool by
hand, so entries written during a session were invisible to ``search_context``
until then. Two triggers fix that:

1. A periodic background sweep (modeled on :mod:`skill_hub.continuous_sweep`):
   staleness-gated (source-file mtimes vs the last run), pressure-gated
   (IDLE only), and skipped entirely when no embedding backend is usable.
2. A task-close refresh: closing a task re-embeds user/plugin memory files
   (idempotent upserts) and reindexes the wiki when its pages changed, so the
   next search sees this task's output.

State (last-run timestamp) persists to
``~/.claude/mcp-skill-hub/state/reindex_sweep.json``.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from . import config as _cfg
from .memory_routing import selected_memory_backend

log = logging.getLogger(__name__)

_DEFAULT_INTERVAL_MINUTES = 1440
_POLL_INTERVAL_SECONDS = 300
_STATE_FILE = Path.home() / ".claude" / "mcp-skill-hub" / "state" / "reindex_sweep.json"

_sweep_timer: threading.Timer | None = None
_sweep_lock = threading.Lock()
_refresh_lock = threading.Lock()   # one refresh at a time (task-close bursts)


def _read_state(state_file: Path | None = None) -> dict:
    path = state_file or _STATE_FILE
    try:
        state = json.loads(path.read_text())
        return state if isinstance(state, dict) else {}
    except (OSError, json.JSONDecodeError, ValueError, KeyError):
        return {}


def _read_last_run(state_file: Path | None = None) -> float:
    try:
        return float(_read_state(state_file).get("last_run", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _write_last_run(ts: float, state_file: Path | None = None,
                    backend: str | None = None) -> None:
    path = state_file or _STATE_FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"last_run": ts}
        if backend in {"raw", "wiki"}:
            state["memory_retrieval_backend"] = backend
        path.write_text(json.dumps(state))
    except OSError as exc:
        log.warning("reindex_sweep: could not write state: %s", exc)


def _wiki_root() -> Path:
    return Path(str(
        _cfg.get("wiki_root")
        or Path.home() / ".claude" / "mcp-skill-hub" / "wiki"
    )).expanduser()


def wiki_stale_since(ts: float, wiki_root: Path | None = None) -> bool:
    """True when any wiki page file changed after ``ts``."""
    root = wiki_root or _wiki_root()
    for sub in ("pages", "_private"):
        base = root / sub
        if not base.is_dir():
            continue
        try:
            for f in base.rglob("*.md"):
                try:
                    if f.stat().st_mtime > ts:
                        return True
                except OSError:
                    continue
        except OSError:
            continue
    return False


def run_refresh(store, *, wiki: bool | None = None,
                state_file: Path | None = None) -> dict:
    """One refresh pass: re-embed memory files, reindex the wiki when stale.

    ``wiki=None`` (default) reindexes only when page files changed since the
    last recorded run; ``True`` forces it; ``False`` skips it. Returns counts.
    Never raises — callers run this from daemon threads.
    """
    result: dict = {}
    # Context digests (#135) need the chat ladder, not embeddings — build them
    # even when no embed backend is up.
    try:
        from .compression.digest import refresh_pending
        built = refresh_pending(store, limit=20)
        if built:
            result["digests_built"] = built
    except Exception as exc:  # noqa: BLE001
        log.debug("reindex_sweep: digest refresh failed: %s", exc)

    from .embeddings import embed_available
    if not embed_available():
        result["skipped"] = "no embedding backend available"
        return result

    with _refresh_lock:
        state = _read_state(state_file)
        try:
            last_run = float(state.get("last_run", 0.0))
        except (TypeError, ValueError):
            last_run = 0.0
        backend = selected_memory_backend(_cfg)
        do_wiki = (
            backend == "wiki"
            and wiki is not False
            and (wiki is True
                 or state.get("memory_retrieval_backend") != "wiki"
                 or wiki_stale_since(last_run))
        )
        if do_wiki:
            try:
                from . import wiki as _wiki_mod
                counts = _wiki_mod.reindex(store, _wiki_root())
                result["wiki_pages"] = counts.get("pages", 0)
                result["wiki_vectors"] = counts.get("vectors", 0)
                if counts.get("errors", 0):
                    result["wiki_error"] = f"{counts['errors']} wiki indexing errors"
            except Exception as exc:  # noqa: BLE001
                log.warning("reindex_sweep: wiki reindex failed: %s", exc)
                result["wiki_error"] = str(exc)[:200]
        try:
            from .memory_index import index_plugin_memory, index_user_memory
            if backend == "raw":
                result["user_memory_files"] = index_user_memory(store)
            elif backend is None:
                result["memory_backend_error"] = (
                    "memory_retrieval_backend must be 'raw' or 'wiki'; "
                    "raw and wiki indexing were skipped"
                )
            if backend == "raw":
                plugin_counts = index_plugin_memory(store)
                result["plugin_memory_files"] = sum(plugin_counts.values())
        except Exception as exc:  # noqa: BLE001
            log.warning("reindex_sweep: memory reindex failed: %s", exc)
            result["memory_error"] = str(exc)[:200]
        if (backend is not None
                and not (backend == "wiki" and (
                    wiki is False or (do_wiki and "wiki_error" in result)
                ))):
            _write_last_run(time.time(), state_file, backend=backend)
    return result


def refresh_after_task_close(store, task_id: int | None = None) -> None:
    """Fire-and-forget refresh after ``close_task`` (daemon thread)."""
    if not _cfg.get("reindex_on_task_close"):
        return

    def _run() -> None:
        try:
            result = run_refresh(store)
            log.info("reindex_sweep: task-close refresh (#%s): %s",
                     task_id, result)
        except Exception as exc:  # noqa: BLE001
            log.warning("reindex_sweep: task-close refresh failed: %s", exc)

    t = threading.Thread(target=_run, name="reindex-task-close", daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Periodic sweep (poll-timer pattern shared with continuous_sweep)
# ---------------------------------------------------------------------------

def _run_sweep(state_file: Path | None = None, _reschedule: bool = True) -> None:
    try:
        if not _cfg.get("reindex_sweep_enabled"):
            return

        interval_minutes = int(
            _cfg.get("reindex_sweep_interval_minutes") or _DEFAULT_INTERVAL_MINUTES
        )
        last_run = _read_last_run(state_file)
        now = time.time()
        if now - last_run < interval_minutes * 60:
            return

        from .resource_monitor import Pressure, snapshot as _snapshot
        if _snapshot().pressure != Pressure.IDLE:
            log.debug("reindex_sweep: skipping — machine not idle")
            return

        # Backend transitions can require a wiki rebuild even when no source
        # page changed, so let run_refresh inspect freshness and backend state.
        if not wiki_stale_since(last_run):
            from .store import get_store
            run_refresh(get_store(), state_file=state_file)
            return

        from .store import get_store
        result = run_refresh(get_store(), state_file=state_file)
        log.info("reindex_sweep: periodic refresh: %s", result)

    except Exception as exc:  # noqa: BLE001
        log.warning("reindex_sweep: unhandled error in sweep tick: %s", exc)
    finally:
        if _reschedule:
            _schedule_next(state_file=state_file)


def _schedule_next(state_file: Path | None = None) -> None:
    if not _cfg.get("reindex_sweep_enabled"):
        return
    global _sweep_timer
    with _sweep_lock:
        t = threading.Timer(
            _POLL_INTERVAL_SECONDS, _run_sweep, kwargs={"state_file": state_file},
        )
        t.daemon = True
        t.name = "reindex-sweep"
        _sweep_timer = t
        t.start()


def start(state_file: Path | None = None) -> None:
    """Start the periodic sweep timer (idempotent; no-op when disabled)."""
    if not _cfg.get("reindex_sweep_enabled"):
        return
    global _sweep_timer
    with _sweep_lock:
        if _sweep_timer is not None and _sweep_timer.is_alive():
            return
    log.info("reindex_sweep: starting (interval=%s min)",
             _cfg.get("reindex_sweep_interval_minutes") or _DEFAULT_INTERVAL_MINUTES)
    _schedule_next(state_file=state_file)
