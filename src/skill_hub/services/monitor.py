"""System resource monitor — drives pressure suggestions and auto-disable.

Uses ``psutil`` when available; falls back to ``os`` helpers otherwise so the
module works even if the optional dep is missing (pressure stays False).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class ResourceSample:
    ram_free_mb: int
    ram_total_mb: int
    cpu_load_1m: float
    cpu_count: int
    pressure: bool
    timestamp: float


def _psutil() -> Any | None:
    try:
        import psutil  # type: ignore

        return psutil
    except ImportError:
        return None


def sample(cfg: dict | None = None) -> ResourceSample:
    """Take a single resource reading."""
    cfg = cfg or {}
    mon_cfg = (cfg.get("monitor") or {})
    ram_threshold_mb = int(mon_cfg.get("ram_free_mb_min") or 2048)
    load_threshold_pct = float(mon_cfg.get("cpu_load_pct_max") or 0.80)

    ps = _psutil()
    if ps is not None:
        vm = ps.virtual_memory()
        ram_free_mb = int(vm.available / 1024 / 1024)
        ram_total_mb = int(vm.total / 1024 / 1024)
        cpu_count = ps.cpu_count(logical=True) or os.cpu_count() or 1
        try:
            cpu_load_1m = ps.getloadavg()[0]  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            cpu_load_1m = 0.0
    else:
        # Fallback — rough values, pressure will stay False unless very low.
        try:
            cpu_load_1m = os.getloadavg()[0]
        except (AttributeError, OSError):
            cpu_load_1m = 0.0
        cpu_count = os.cpu_count() or 1
        ram_free_mb = 0
        ram_total_mb = 0

    pressure = False
    if ram_total_mb and ram_free_mb < ram_threshold_mb:
        pressure = True
    if cpu_count and cpu_load_1m / cpu_count > load_threshold_pct:
        pressure = True

    return ResourceSample(
        ram_free_mb=ram_free_mb,
        ram_total_mb=ram_total_mb,
        cpu_load_1m=cpu_load_1m,
        cpu_count=cpu_count,
        pressure=pressure,
        timestamp=time.time(),
    )


class PressureTracker:
    """Smooths pressure over time so :meth:`sustained_seconds` avoids flapping."""

    def __init__(self, load_config_callable=None) -> None:
        self._load_cfg = load_config_callable or (lambda: {})
        self._since: float | None = None
        self._last: ResourceSample | None = None

    def sample(self) -> ResourceSample:
        s = sample(self._load_cfg())
        now = s.timestamp
        if s.pressure:
            if self._since is None:
                self._since = now
        else:
            self._since = None
        self._last = s
        return s

    def sustained_seconds(self) -> float:
        if self._since is None:
            return 0.0
        return max(0.0, time.time() - self._since)

    def last_sample(self) -> ResourceSample | None:
        return self._last
