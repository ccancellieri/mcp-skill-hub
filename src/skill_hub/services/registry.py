"""Service registry — single source of truth for managed services.

The registry owns Service instances, reconciles desired-vs-actual state from
config, and exposes the list of currently-disabled services + resource
suggestions to the webapp.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from .base import Service
from .inproc import HaikuRouter, WatcherService
from .ollama import OllamaDaemon, OllamaModel
from .searxng import SearxngContainer

if TYPE_CHECKING:
    from .monitor import PressureTracker, ResourceSample

log = logging.getLogger(__name__)


_SINGLETON: "ServiceRegistry | None" = None
_SINGLETON_PRESSURE: "PressureTracker | None" = None


def get_registry() -> "ServiceRegistry":
    """Return the process-wide registry, building it from config on first call."""
    global _SINGLETON
    if _SINGLETON is None:
        from .. import config as _cfg
        _SINGLETON = ServiceRegistry.build_from_config(_cfg.load_config())
    return _SINGLETON


def set_registry(registry: "ServiceRegistry") -> None:
    """Install a pre-built registry (called by server.py during startup)."""
    global _SINGLETON
    _SINGLETON = registry


def get_pressure() -> "PressureTracker":
    global _SINGLETON_PRESSURE
    if _SINGLETON_PRESSURE is None:
        from .monitor import PressureTracker
        from .. import config as _cfg
        _SINGLETON_PRESSURE = PressureTracker(load_config_callable=_cfg.load_config)
    return _SINGLETON_PRESSURE


def set_pressure(pressure: "PressureTracker") -> None:
    global _SINGLETON_PRESSURE
    _SINGLETON_PRESSURE = pressure


_DEFAULT_SERVICES = (
    "ollama_daemon",
    "ollama_router",
    "ollama_embed",
    "searxng",
    "watcher",
    "haiku_router",
)


def _service_cfg(cfg: dict, name: str) -> dict:
    return ((cfg.get("services") or {}).get(name)) or {}


def _build_instance(name: str, svc_cfg: dict) -> Service:
    if name == "ollama_daemon":
        return OllamaDaemon()
    if name == "ollama_router":
        return OllamaModel(
            name="ollama_router",
            label="Ollama router model",
            description="Local LLM used by router tier-2 classification.",
            model=str(svc_cfg.get("model") or "qwen2.5:3b"),
            approx_ram_mb=int(svc_cfg.get("approx_ram_mb") or 2000),
        )
    if name == "ollama_embed":
        return OllamaModel(
            name="ollama_embed",
            label="Ollama embedding model",
            description="Local embeddings for skill vector search.",
            model=str(svc_cfg.get("model") or "nomic-embed-text"),
            approx_ram_mb=int(svc_cfg.get("approx_ram_mb") or 500),
        )
    if name == "searxng":
        return SearxngContainer(container=str(svc_cfg.get("container") or "skill-hub-searxng"))
    if name == "watcher":
        return WatcherService()
    if name == "haiku_router":
        return HaikuRouter()
    raise KeyError(f"unknown service: {name}")


class ServiceRegistry:
    """Holds Service instances and drives reconciliation."""

    def __init__(self, services: Iterable[Service]) -> None:
        self._services: dict[str, Service] = {s.name: s for s in services}
        self._disabled: set[str] = set()
        self._last_sample: Any | None = None
        self._suggestions: list[str] = []

    @classmethod
    def build_from_config(cls, cfg: dict) -> "ServiceRegistry":
        services: list[Service] = []
        for name in _DEFAULT_SERVICES:
            services.append(_build_instance(name, _service_cfg(cfg, name)))
        return cls(services)

    # --- accessors -------------------------------------------------------

    def get(self, name: str) -> Service | None:
        return self._services.get(name)

    def all(self) -> list[Service]:
        return list(self._services.values())

    @property
    def disabled_services(self) -> list[str]:
        return sorted(self._services[n].label for n in self._disabled if n in self._services)

    @property
    def last_sample(self) -> Any | None:
        return self._last_sample

    @property
    def suggestions(self) -> list[str]:
        return list(self._suggestions)

    # --- reconcile -------------------------------------------------------

    def apply_mutable_config(self, cfg: dict) -> None:
        """Rebuild services whose mutable fields (model, container) changed.

        Called on every reconcile so that after a form POST updates
        ``services.ollama_router.model`` the next status check uses the new model.
        """
        for name in list(self._services.keys()):
            svc_cfg = _service_cfg(cfg, name)
            current = self._services[name]
            # Only a few services carry mutable fields; skip others.
            if isinstance(current, OllamaModel):
                new_model = str(svc_cfg.get("model") or "")
                if new_model and new_model != current.model:
                    self._services[name] = _build_instance(name, svc_cfg)
            elif isinstance(current, SearxngContainer):
                new_c = str(svc_cfg.get("container") or "")
                if new_c and new_c != current.container:
                    self._services[name] = _build_instance(name, svc_cfg)

    def reconcile(self, cfg: dict, pressure: "PressureTracker | None" = None) -> None:
        """Drive service states to match config; update disabled set + suggestions."""
        self.apply_mutable_config(cfg)

        disabled: set[str] = set()
        for name, svc in self._services.items():
            svc_cfg = _service_cfg(cfg, name)
            desired = bool(svc_cfg.get("enabled", True))
            auto_disable = bool(svc_cfg.get("auto_disable_under_pressure", False))

            if pressure is not None and auto_disable and pressure.sustained_seconds() > 30:
                desired = False

            current = svc.status()

            if desired and current == "stopped":
                ok, msg = svc.start()
                if not ok:
                    log.warning("start %s failed: %s", name, msg)
            elif not desired and current == "running":
                ok, msg = svc.stop()
                if not ok:
                    log.warning("stop %s failed: %s", name, msg)

            if not desired or svc.status() != "running":
                disabled.add(name)

        self._disabled = disabled
        self._suggestions = self._compute_suggestions(cfg, pressure)

    def startup_align(self, cfg: dict) -> None:
        """Honour ``auto_start`` on first boot.

        Services with ``enabled=True and auto_start=True`` are started;
        ``enabled=False`` services are stopped to align OS state.
        """
        for name, svc in self._services.items():
            svc_cfg = _service_cfg(cfg, name)
            enabled = bool(svc_cfg.get("enabled", True))
            auto_start = bool(svc_cfg.get("auto_start", True))
            current = svc.status()
            if enabled and auto_start and current == "stopped":
                ok, msg = svc.start()
                if not ok:
                    log.info("auto_start %s skipped: %s", name, msg)
            elif not enabled and current == "running":
                ok, msg = svc.stop()
                if not ok:
                    log.info("auto_stop %s skipped: %s", name, msg)

    def record_sample(self, sample: Any) -> None:
        self._last_sample = sample

    # --- suggestions -----------------------------------------------------

    def _compute_suggestions(self, cfg: dict, pressure: "PressureTracker | None") -> list[str]:
        if pressure is None:
            return []
        sample = pressure.last_sample()
        if sample is None:
            return []
        suggestions: list[str] = []
        if sample.pressure:
            running = [
                (n, s) for n, s in self._services.items() if s.status() == "running"
            ]
            running.sort(
                key=lambda pair: pair[1].resource_footprint().get("ram_mb_approx", 0),
                reverse=True,
            )
            for name, svc in running[:2]:
                fp = svc.resource_footprint()
                suggestions.append(
                    f"Consider stopping {svc.label} (~{fp['ram_mb_approx']} MB) — free RAM "
                    f"{sample.ram_free_mb} MB"
                )
        return suggestions


# ---------------------------------------------------------------------------
# Reconciler daemon thread
# ---------------------------------------------------------------------------


class ReconcilerHandle:
    """Stop-handle returned by :func:`start_reconciler`."""

    def __init__(self, stop_event: threading.Event, thread: threading.Thread) -> None:
        self._stop = stop_event
        self._thread = thread

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def start_reconciler(
    registry: ServiceRegistry,
    pressure: "PressureTracker",
    config_path: Path,
    load_config,  # callable returning the fresh config dict
    interval_sec: float = 2.0,
) -> ReconcilerHandle:
    """Spawn a daemon thread that reconciles state every ``interval_sec``."""
    stop_event = threading.Event()
    last_mtime = [0.0]

    # Align OS state to config at startup.
    try:
        registry.startup_align(load_config())
    except Exception as e:  # noqa: BLE001
        log.warning("startup_align failed: %s", e)

    def _loop() -> None:
        while not stop_event.is_set():
            try:
                sample = pressure.sample()
                registry.record_sample(sample)

                try:
                    mtime = config_path.stat().st_mtime
                except OSError:
                    mtime = 0.0

                cfg_changed = mtime != last_mtime[0]
                last_mtime[0] = mtime

                cfg = load_config()
                if cfg_changed or True:
                    registry.reconcile(cfg, pressure)
            except Exception as e:  # noqa: BLE001
                log.exception("reconciler tick failed: %s", e)
            stop_event.wait(interval_sec)

    thread = threading.Thread(target=_loop, name="skill-hub-reconciler", daemon=True)
    thread.start()
    return ReconcilerHandle(stop_event, thread)
