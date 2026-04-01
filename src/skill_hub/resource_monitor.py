"""Cross-platform resource monitor — adapts LLM operations to system pressure.

Detects CPU load and memory pressure on macOS, Linux, and WSL.
Returns a pressure level that the hook pipeline uses to decide how
aggressively to run local LLM operations (triage, pre-compaction,
digests, background memory optimization).

Pressure levels:
  idle     — machine barely used, ideal for background LLM work
  low      — light use, all LLM operations run normally
  moderate — noticeable load, skip optional LLM calls (digest, precompact)
  high     — heavy load, skip all LLM calls, pass through to Claude
"""

import logging
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from enum import IntEnum

log = logging.getLogger(__name__)


class Pressure(IntEnum):
    """System pressure levels — higher = more constrained."""
    IDLE = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3


@dataclass
class SystemSnapshot:
    """Point-in-time system resource snapshot."""
    pressure: Pressure
    cpu_load_1m: float       # 1-minute load average (normalized to core count)
    memory_used_pct: float   # 0.0–1.0
    memory_available_mb: int
    total_memory_mb: int
    timestamp: float

    def __str__(self) -> str:
        return (
            f"pressure={self.pressure.name} "
            f"cpu={self.cpu_load_1m:.1%} "
            f"mem={self.memory_used_pct:.0%} "
            f"avail={self.memory_available_mb}MB"
        )


# Thresholds — tuned for machines running Ollama alongside other work
_THRESHOLDS = {
    # (cpu_load_normalized, memory_used_pct) → Pressure
    # Checked in order: first match wins
    "high":     {"cpu": 0.85, "mem": 0.90},
    "moderate": {"cpu": 0.60, "mem": 0.80},
    "low":      {"cpu": 0.30, "mem": 0.60},
    # below "low" = idle
}

# Cache: avoid checking system state on every message
_cache: SystemSnapshot | None = None
_CACHE_TTL_SECONDS = 10.0


def _get_cpu_count() -> int:
    """Get logical CPU count."""
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


def _get_load_avg() -> tuple[float, float, float]:
    """Get 1/5/15-minute load averages. Cross-platform."""
    try:
        return os.getloadavg()
    except (OSError, AttributeError):
        # Windows/WSL fallback
        pass

    # Try /proc/loadavg on Linux/WSL
    try:
        with open("/proc/loadavg") as f:
            parts = f.read().split()
            return float(parts[0]), float(parts[1]), float(parts[2])
    except (FileNotFoundError, ValueError, IndexError):
        pass

    return (0.0, 0.0, 0.0)


def _get_memory_darwin() -> tuple[int, int]:
    """macOS: return (available_mb, total_mb)."""
    try:
        import ctypes
        import ctypes.util
        libc = ctypes.CDLL(ctypes.util.find_library("c"))

        # Total memory via sysctl
        total = ctypes.c_int64()
        size = ctypes.c_size_t(8)
        libc.sysctlbyname(b"hw.memsize", ctypes.byref(total),
                          ctypes.byref(size), None, 0)
        total_mb = total.value // (1024 * 1024)

        # Available = free + inactive (macOS compresses inactive)
        result = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=5
        )
        pages = {}
        for line in result.stdout.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                val = val.strip().rstrip(".")
                try:
                    pages[key.strip()] = int(val)
                except ValueError:
                    pass

        page_size = 16384  # Apple Silicon default
        free_pages = pages.get("Pages free", 0)
        inactive_pages = pages.get("Pages inactive", 0)
        available_mb = (free_pages + inactive_pages) * page_size // (1024 * 1024)

        return available_mb, total_mb
    except Exception:
        return 8192, 16384  # safe fallback


def _get_memory_linux() -> tuple[int, int]:
    """Linux: return (available_mb, total_mb)."""
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                if ":" in line:
                    key, val = line.split(":", 1)
                    # Value is in kB
                    val = val.strip().split()[0]
                    try:
                        info[key.strip()] = int(val)
                    except ValueError:
                        pass

            total_kb = info.get("MemTotal", 16 * 1024 * 1024)
            available_kb = info.get("MemAvailable", total_kb // 2)
            return available_kb // 1024, total_kb // 1024
    except (FileNotFoundError, ValueError):
        return 8192, 16384


def _get_memory() -> tuple[int, int]:
    """Cross-platform: return (available_mb, total_mb)."""
    system = platform.system()
    if system == "Darwin":
        return _get_memory_darwin()
    elif system == "Linux":
        return _get_memory_linux()
    else:
        return 8192, 16384  # fallback


def snapshot(force: bool = False) -> SystemSnapshot:
    """Get current system resource snapshot (cached for 10s)."""
    global _cache

    now = time.monotonic()
    if not force and _cache and (now - _cache.timestamp) < _CACHE_TTL_SECONDS:
        return _cache

    cpu_count = _get_cpu_count()
    load_1m, _, _ = _get_load_avg()
    cpu_normalized = load_1m / cpu_count  # 0.0 = idle, 1.0 = all cores at 100%

    available_mb, total_mb = _get_memory()
    mem_used_pct = 1.0 - (available_mb / total_mb) if total_mb > 0 else 0.5

    # Classify pressure
    if (cpu_normalized >= _THRESHOLDS["high"]["cpu"]
            or mem_used_pct >= _THRESHOLDS["high"]["mem"]):
        pressure = Pressure.HIGH
    elif (cpu_normalized >= _THRESHOLDS["moderate"]["cpu"]
          or mem_used_pct >= _THRESHOLDS["moderate"]["mem"]):
        pressure = Pressure.MODERATE
    elif (cpu_normalized >= _THRESHOLDS["low"]["cpu"]
          or mem_used_pct >= _THRESHOLDS["low"]["mem"]):
        pressure = Pressure.LOW
    else:
        pressure = Pressure.IDLE

    _cache = SystemSnapshot(
        pressure=pressure,
        cpu_load_1m=cpu_normalized,
        memory_used_pct=mem_used_pct,
        memory_available_mb=available_mb,
        total_memory_mb=total_mb,
        timestamp=now,
    )

    log.debug("Resource snapshot: %s", _cache)
    return _cache


def should_run_llm(operation: str) -> bool:
    """Decide whether an LLM operation should run given current pressure.

    Operations and their pressure tolerance:
      triage          — runs up to MODERATE (core routing, skip only under HIGH)
      precompact      — runs up to LOW (optional optimization, skip early)
      digest          — runs up to LOW (periodic, deferrable)
      rerank          — runs up to MODERATE (improves quality, not critical)
      optimize_memory — runs only at IDLE (expensive, background-only)
      embed           — always runs (fast, ~100ms, needed for search)

    Returns True (always run) if resource_gating_enabled is False in config.
    Also returns True if SKILL_HUB_FORCE_LLM=1 env var is set (manual override).
    """
    # Config gate: allow disabling resource awareness entirely
    from . import config as _cfg
    if not _cfg.get("resource_gating_enabled"):
        return True

    # Env override: force all LLM ops to run (for manual/cron invocations)
    if os.environ.get("SKILL_HUB_FORCE_LLM") == "1":
        return True

    s = snapshot()

    limits: dict[str, Pressure] = {
        "triage":          Pressure.MODERATE,
        "precompact":      Pressure.LOW,
        "digest":          Pressure.LOW,
        "rerank":          Pressure.MODERATE,
        "optimize_memory": Pressure.IDLE,
        "embed":           Pressure.HIGH,  # always runs
    }

    max_pressure = limits.get(operation, Pressure.LOW)
    allowed = s.pressure <= max_pressure

    if not allowed:
        log.info(
            "Skipping %s: pressure=%s > limit=%s (cpu=%.0f%% mem=%.0f%%)",
            operation, s.pressure.name, max_pressure.name,
            s.cpu_load_1m * 100, s.memory_used_pct * 100,
        )

    return allowed
