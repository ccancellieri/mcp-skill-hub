"""Filesystem watcher for auto-reindex of skills and plugins.

When a .json or .md file changes inside local_skills_dir or any plugin
directory, the watcher debounces the event and triggers a full reindex
so the skill store stays fresh without manual /hub-index-skills calls.

Optional dependency: watchdog>=4.0.0
If watchdog is not installed, start_watcher() returns None silently.
"""

from __future__ import annotations

import threading
from pathlib import Path


def start_watcher() -> object | None:
    """
    Start a debounced Observer watching local_skills_dir and plugin dirs.

    Returns:
        Running watchdog.Observer instance, or None if watchdog is not installed.
    """
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        return None

    from . import config as _cfg

    handler = _DebounceHandler()

    observer = Observer()

    # Watch local skills dir
    skills_dir = Path(str(_cfg.get("local_skills_dir") or
                          Path.home() / ".claude" / "local-skills")).expanduser()
    if skills_dir.exists():
        observer.schedule(handler, str(skills_dir), recursive=False)

    # Watch extra skill dirs
    for entry in (_cfg.get("extra_skill_dirs") or []):
        p = Path(str(entry.get("path", ""))).expanduser()
        if p.exists() and entry.get("enabled", True):
            observer.schedule(handler, str(p), recursive=False)

    # Watch plugin dirs via indexer
    try:
        from .indexer import PLUGIN_DIRS
        for pdir in PLUGIN_DIRS:
            p = Path(pdir).expanduser()
            if p.exists():
                observer.schedule(handler, str(p), recursive=True)
    except Exception:
        pass

    observer.start()
    return observer


def stop_watcher(observer: object) -> None:
    """Stop the observer gracefully."""
    try:
        observer.stop()  # type: ignore[attr-defined]
        observer.join(timeout=3)
    except Exception:
        pass


_IGNORE_PATH_PARTS = frozenset({"temp_git_", "__pycache__", ".git", "node_modules"})

# Minimum seconds between two completed reindex runs.
# Prevents back-to-back reindexes when plugin manager creates many temp dirs.
_MIN_REINDEX_INTERVAL = 120.0


class _DebounceHandler:
    """
    Filesystem event handler with debounce + post-completion cooldown.

    - Debounce: waits `delay` seconds of silence before triggering.
    - Cooldown: ignores new events for `_MIN_REINDEX_INTERVAL` seconds
      after a reindex completes — suppresses the storm of temp_git_* events
      that Claude Code's plugin manager generates.
    - Skips paths containing known temp/internal dir names.
    """

    def __init__(self, delay: float = 2.0) -> None:
        self._delay = delay
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._last_changed: str = ""
        self._reindexing: bool = False
        self._last_reindex_done: float = 0.0  # monotonic time of last completion

    def dispatch(self, event: object) -> None:
        src = getattr(event, "src_path", "")
        if not (src.endswith(".json") or src.endswith(".md")):
            return
        # Ignore temp/internal directories
        if any(part in src for part in _IGNORE_PATH_PARTS):
            return
        with self._lock:
            import time as _time
            # If a reindex is running, or cooldown hasn't expired, skip
            if self._reindexing:
                return
            if (_time.monotonic() - self._last_reindex_done) < _MIN_REINDEX_INTERVAL:
                return
            self._last_changed = src
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._delay, self._do_reindex)
            self._timer.daemon = True
            self._timer.start()

    def _do_reindex(self) -> None:
        import time
        with self._lock:
            if self._reindexing:
                return
            self._reindexing = True

        try:
            from .activity_log import log_event
            changed_name = Path(self._last_changed).name if self._last_changed else "unknown"
            log_event("WATCHER", f"re-indexing skills (trigger: {changed_name})")

            from .indexer import index_all
            from .store import SkillStore
            t0 = time.monotonic()
            _store = SkillStore()
            try:
                count, errors = index_all(_store)
            finally:
                _store.close()
            elapsed = time.monotonic() - t0

            # Invalidate the Level 3 skill cache in cli.py
            try:
                from . import cli as _cli
                _cli._local_skills_cache = None
                _cli._local_skills_hash = None
            except Exception:
                pass

            err_msg = f"  errors={len(errors)}" if errors else ""
            log_event("WATCHER", f"re-index complete: {count} skills in {elapsed:.1f}s{err_msg}")
        except Exception as exc:
            try:
                from .activity_log import log_event
                log_event("WATCHER", f"re-index failed: {exc}")
            except Exception:
                pass
        finally:
            with self._lock:
                self._reindexing = False
                self._last_reindex_done = time.monotonic()
