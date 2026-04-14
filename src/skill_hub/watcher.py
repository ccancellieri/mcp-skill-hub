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

    # Plugin extension-point: A9 — register per-plugin watch_paths.
    # Each enabled plugin's plugin.json may declare `watch_paths: [glob, ...]`
    # relative to the plugin root. On change we dispatch to the plugin's
    # indexer.index_doc(store, path) if present, else a full index_docs(store).
    try:
        from .plugin_registry import iter_enabled_plugins
        for pinfo in iter_enabled_plugins():
            globs = pinfo["manifest"].get("watch_paths") or []
            if not globs:
                continue
            watched_dirs: set[Path] = set()
            for g in globs:
                for match in pinfo["path"].glob(g):
                    d = match.parent if match.is_file() else match
                    watched_dirs.add(d)
                # Also watch the nearest existing ancestor of the glob pattern
                # so files added later get caught.
                head = str(g).split("*", 1)[0]
                if head:
                    anchor = (pinfo["path"] / head).resolve()
                    while not anchor.exists() and anchor != anchor.parent:
                        anchor = anchor.parent
                    if anchor.exists():
                        watched_dirs.add(anchor)
            for d in watched_dirs:
                if d.exists() and d.is_dir():
                    observer.schedule(
                        _PluginPathHandler(pinfo),
                        str(d),
                        recursive=True,
                    )
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
        self._pending_paths: set[Path] = set()
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
            try:
                self._pending_paths.add(Path(src).resolve())
            except OSError:
                self._pending_paths.add(Path(src))
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
            paths = set(self._pending_paths)
            self._pending_paths.clear()

        try:
            from .activity_log import log_event
            changed_name = Path(self._last_changed).name if self._last_changed else "unknown"
            log_event("WATCHER",
                      f"re-indexing skills (trigger: {changed_name}, "
                      f"paths={len(paths)})")

            from .indexer import index_all
            from .store import SkillStore
            t0 = time.monotonic()
            _store = SkillStore()
            try:
                count, errors = index_all(_store,
                                          changed_paths=paths or None)
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


class _PluginPathHandler:
    """Plugin extension-point: A9 — dispatcher for plugin-declared watch_paths.

    Reuses the 2s debounce + 120s cooldown pattern of ``_DebounceHandler`` but
    routes the event to the plugin's ``indexer.index_doc(store, path)`` if
    present, else a full ``indexer.index_docs(store)``.
    """

    def __init__(self, plugin_info: dict, delay: float = 2.0) -> None:
        self._plugin = plugin_info
        self._delay = delay
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._last_path: str = ""
        self._busy: bool = False
        self._last_done: float = 0.0

    def dispatch(self, event: object) -> None:
        src = getattr(event, "src_path", "")
        if not (src.endswith(".json") or src.endswith(".md")):
            return
        if any(part in src for part in _IGNORE_PATH_PARTS):
            return
        with self._lock:
            import time as _time
            if self._busy:
                return
            if (_time.monotonic() - self._last_done) < _MIN_REINDEX_INTERVAL:
                return
            self._last_path = src
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._delay, self._dispatch_plugin_indexer)
            self._timer.daemon = True
            self._timer.start()

    def _dispatch_plugin_indexer(self) -> None:
        import importlib.util
        import sys
        import time

        with self._lock:
            if self._busy:
                return
            self._busy = True

        try:
            from .activity_log import log_event
            from .store import SkillStore

            plugin_path = Path(self._plugin["path"])
            indexer_rel = self._plugin["manifest"].get("indexer", "indexer.py")
            indexer_py = plugin_path / indexer_rel
            if not indexer_py.exists():
                return

            mod_name = f"_skillhub_plugin_{plugin_path.name}_indexer"
            spec = importlib.util.spec_from_file_location(mod_name, indexer_py)
            if spec is None or spec.loader is None:
                return
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            try:
                spec.loader.exec_module(mod)
            except Exception as exc:  # noqa: BLE001
                log_event("WATCHER", f"plugin indexer import failed "
                          f"({self._plugin['name']}): {exc}")
                return

            t0 = time.monotonic()
            store = SkillStore()
            try:
                index_doc = getattr(mod, "index_doc", None)
                index_docs = getattr(mod, "index_docs", None)
                if callable(index_doc) and self._last_path:
                    index_doc(store, Path(self._last_path))
                elif callable(index_docs):
                    index_docs(store)
            finally:
                store.close()
            elapsed = time.monotonic() - t0
            log_event("WATCHER",
                      f"plugin {self._plugin['name']}: re-indexed in "
                      f"{elapsed:.1f}s (trigger: {Path(self._last_path).name})")
        except Exception as exc:  # noqa: BLE001
            try:
                from .activity_log import log_event
                log_event("WATCHER", f"plugin re-index failed: {exc}")
            except Exception:
                pass
        finally:
            import time as _time
            with self._lock:
                self._busy = False
                self._last_done = _time.monotonic()
