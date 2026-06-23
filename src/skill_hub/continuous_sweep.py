"""Continuous memory compaction sweep — background, pressure-gated, idempotent.

Runs ``promote_memory(dry_run=False)`` on a configurable interval only when
the machine is IDLE (Pressure.IDLE).  Enabled via ``continuous_sweep_enabled``
in config (default False).

State (last-run timestamp) is persisted to
``~/.claude/mcp-skill-hub/state/continuous_sweep.json`` so the guard works
across restarts and cooperates with the weekly cron / postcompact paths.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable

from . import config as _cfg
from .resource_monitor import Pressure, snapshot as _snapshot

log = logging.getLogger(__name__)

_DEFAULT_INTERVAL_MINUTES = 60
_STATE_FILE = Path.home() / ".claude" / "mcp-skill-hub" / "state" / "continuous_sweep.json"

# Module-level handle so tests can inspect / replace the thread.
_sweep_thread: threading.Timer | None = None
_sweep_lock = threading.Lock()


def _state_path() -> Path:
    return _STATE_FILE


def _read_last_run(state_file: Path | None = None) -> float:
    """Return the epoch timestamp of the last completed sweep (0.0 if never)."""
    path = state_file or _state_path()
    try:
        data = json.loads(path.read_text())
        return float(data.get("last_run", 0.0))
    except (OSError, json.JSONDecodeError, ValueError, KeyError):
        return 0.0


def _write_last_run(ts: float, state_file: Path | None = None) -> None:
    """Persist the last-run timestamp."""
    path = state_file or _state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"last_run": ts}))
    except OSError as exc:
        log.warning("continuous_sweep: could not write state: %s", exc)


def _run_sweep(
    promote_fn: Callable[..., str] | None = None,
    state_file: Path | None = None,
    _interval_minutes: int | None = None,
    _reschedule: bool = True,
) -> None:
    """Execute one sweep tick: pressure-check, interval-check, then promote.

    All arguments are injectable for testing; production callers use defaults.
    """
    try:
        if not _cfg.get("continuous_sweep_enabled"):
            # Flag was toggled off after the timer fired — do not reschedule.
            log.debug("continuous_sweep: disabled, exiting sweep loop")
            return

        interval_minutes = _interval_minutes if _interval_minutes is not None else int(
            _cfg.get("continuous_sweep_interval_minutes") or _DEFAULT_INTERVAL_MINUTES
        )
        interval_seconds = interval_minutes * 60

        last_run = _read_last_run(state_file)
        now = time.time()

        if now - last_run < interval_seconds:
            remaining = interval_seconds - (now - last_run)
            log.debug(
                "continuous_sweep: skipping — %.0f s remaining until next window",
                remaining,
            )
        else:
            snap = _snapshot()
            if snap.pressure != Pressure.IDLE:
                log.info(
                    "continuous_sweep: skipping — pressure=%s (need IDLE)",
                    snap.pressure.name,
                )
            else:
                log.info("continuous_sweep: running promote_memory (pressure=IDLE)")
                try:
                    if promote_fn is not None:
                        promote_fn(dry_run=False)
                    else:
                        # Deferred import to avoid circular dependency at module load.
                        from .server import promote_memory as _promote
                        _promote(dry_run=False)
                    _write_last_run(time.time(), state_file)
                    log.info("continuous_sweep: promote_memory completed")
                except Exception as exc:  # noqa: BLE001
                    log.warning("continuous_sweep: promote_memory raised: %s", exc)

    except Exception as exc:  # noqa: BLE001
        # Never propagate into the timer thread — log and continue.
        log.warning("continuous_sweep: unhandled error in sweep tick: %s", exc)

    finally:
        if _reschedule:
            _schedule_next(promote_fn=promote_fn, state_file=state_file)


def _schedule_next(
    promote_fn: Callable[..., str] | None = None,
    state_file: Path | None = None,
) -> None:
    """Re-arm the one-shot Timer for the next tick (every 5 minutes)."""
    if not _cfg.get("continuous_sweep_enabled"):
        return

    global _sweep_thread
    # Check interval every 5 minutes regardless of the configured sweep interval.
    # This keeps the timer responsive to config changes without waking too often.
    _POLL_INTERVAL_SECONDS = 300

    with _sweep_lock:
        t = threading.Timer(
            _POLL_INTERVAL_SECONDS,
            _run_sweep,
            kwargs={"promote_fn": promote_fn, "state_file": state_file},
        )
        t.daemon = True
        t.name = "continuous-memory-sweep"
        _sweep_thread = t
        t.start()


def start(
    promote_fn: Callable[..., str] | None = None,
    state_file: Path | None = None,
) -> None:
    """Start the background sweep timer (idempotent — safe to call multiple times).

    Called once at server startup when ``continuous_sweep_enabled`` is True.
    Does nothing (and never raises) when the flag is off.
    """
    if not _cfg.get("continuous_sweep_enabled"):
        return

    global _sweep_thread
    with _sweep_lock:
        if _sweep_thread is not None and _sweep_thread.is_alive():
            log.debug("continuous_sweep: already running, ignoring duplicate start()")
            return

    log.info("continuous_sweep: starting background sweep (interval=%s min)",
             _cfg.get("continuous_sweep_interval_minutes") or _DEFAULT_INTERVAL_MINUTES)

    # Fire first tick after the poll interval (don't compete with server startup).
    _schedule_next(promote_fn=promote_fn, state_file=state_file)
