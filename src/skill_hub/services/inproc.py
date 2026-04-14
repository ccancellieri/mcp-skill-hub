"""In-process services — wrap existing asyncio/thread tasks."""
from __future__ import annotations

from threading import Lock

from .base import Service, Status


class WatcherService(Service):
    """Filesystem watcher (watchdog.Observer) for auto-reindex of skills/plugins."""

    name = "watcher"
    label = "File watcher"
    description = "Auto-reindex on skill/plugin file changes."

    def __init__(self) -> None:
        self._observer: object | None = None
        self._lock = Lock()

    def _running(self) -> bool:
        obs = self._observer
        if obs is None:
            return False
        is_alive = getattr(obs, "is_alive", None)
        try:
            return bool(is_alive()) if callable(is_alive) else False
        except Exception:
            return False

    def status(self) -> Status:
        try:
            import watchdog  # noqa: F401
        except ImportError:
            return "unavailable"
        return "running" if self._running() else "stopped"

    def is_available(self) -> tuple[bool, str]:
        try:
            import watchdog  # noqa: F401
            return True, ""
        except ImportError:
            return False, "watchdog not installed — `pip install watchdog>=4.0.0`"

    def start(self) -> tuple[bool, str]:
        from ..watcher import start_watcher

        with self._lock:
            if self._running():
                return True, "already running"
            obs = start_watcher()
            if obs is None:
                return False, "watchdog unavailable"
            self._observer = obs
            return True, "started"

    def stop(self) -> tuple[bool, str]:
        from ..watcher import stop_watcher

        with self._lock:
            obs = self._observer
            if obs is None:
                return True, "already stopped"
            try:
                stop_watcher(obs)
            except Exception as e:  # noqa: BLE001
                return False, f"stop failed: {e}"
            self._observer = None
            return True, "stopped"

    def resource_footprint(self) -> dict:
        return {"ram_mb_approx": 30, "cpu_share": 0.01}


class HaikuRouter(Service):
    """T3 Haiku cloud-API gate — config-only, no process to manage."""

    name = "haiku_router"
    label = "Haiku router (T3)"
    description = "Cloud API tier. Costs money per call. No local process."

    def __init__(self) -> None:
        self._enabled = False

    def status(self) -> Status:
        return "running" if self._enabled else "stopped"

    def is_available(self) -> tuple[bool, str]:
        return True, ""

    def start(self) -> tuple[bool, str]:
        self._enabled = True
        return True, "enabled"

    def stop(self) -> tuple[bool, str]:
        self._enabled = False
        return True, "disabled"

    def set_enabled(self, value: bool) -> None:
        """Sync helper used by the registry to reflect config without a start/stop call."""
        self._enabled = bool(value)
