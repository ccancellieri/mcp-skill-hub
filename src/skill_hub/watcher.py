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


class _DebounceHandler:
    """
    Filesystem event handler with debounce: waits `delay` seconds of
    silence after the last event before triggering reindex.
    Only reacts to .json and .md file events.
    """

    def __init__(self, delay: float = 1.5) -> None:
        self._delay = delay
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    # watchdog calls dispatch() → on_any_event() for all event types
    def dispatch(self, event: object) -> None:
        src = getattr(event, "src_path", "")
        if not (src.endswith(".json") or src.endswith(".md")):
            return
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._delay, self._do_reindex)
            self._timer.daemon = True
            self._timer.start()

    def _do_reindex(self) -> None:
        try:
            from .activity_log import log_event
            log_event("WATCHER", "file change detected — re-indexing skills")

            from .indexer import index_all
            index_all()

            # Invalidate the Level 3 skill cache in cli.py
            try:
                from . import cli as _cli
                _cli._local_skills_cache = None
                _cli._local_skills_hash = None
            except Exception:
                pass

            log_event("WATCHER", "re-index complete")
        except Exception as exc:
            try:
                from .activity_log import log_event
                log_event("WATCHER", f"re-index error: {exc}")
            except Exception:
                pass
