"""Tests for the resource monitor and pressure tracker."""
from __future__ import annotations

import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from skill_hub.services import monitor as mon  # noqa: E402


class _FakeVM:
    def __init__(self, free_mb: int, total_mb: int = 16000):
        self.available = free_mb * 1024 * 1024
        self.total = total_mb * 1024 * 1024


class _FakePsutil:
    def __init__(self, free_mb: int, load: float, cpu_count: int = 8):
        self._free_mb = free_mb
        self._load = load
        self._cpu = cpu_count

    def virtual_memory(self):
        return _FakeVM(self._free_mb)

    def cpu_count(self, logical: bool = True):
        return self._cpu

    def getloadavg(self):
        return (self._load, self._load, self._load)


def _patch_psutil(monkeypatch, free_mb: int, load: float, cpu: int = 8):
    fake = _FakePsutil(free_mb, load, cpu)
    monkeypatch.setattr(mon, "_psutil", lambda: fake)


def test_pressure_false_with_headroom(monkeypatch):
    _patch_psutil(monkeypatch, free_mb=8000, load=1.0)
    s = mon.sample()
    assert s.pressure is False


def test_pressure_true_low_ram(monkeypatch):
    _patch_psutil(monkeypatch, free_mb=1024, load=1.0)
    s = mon.sample()
    assert s.pressure is True


def test_pressure_true_high_load(monkeypatch):
    _patch_psutil(monkeypatch, free_mb=8000, load=10.0, cpu=8)
    s = mon.sample({"monitor": {"cpu_load_pct_max": 0.8}})
    assert s.pressure is True


def test_tracker_sustained_seconds(monkeypatch):
    _patch_psutil(monkeypatch, free_mb=500, load=0.5)
    t = mon.PressureTracker()
    t.sample()
    # Force the "since" to be 5 seconds ago to simulate elapsed time.
    t._since = time.time() - 5
    assert 4 < t.sustained_seconds() < 7


def test_tracker_resets_when_pressure_clears(monkeypatch):
    _patch_psutil(monkeypatch, free_mb=500, load=0.5)
    t = mon.PressureTracker()
    t.sample()
    assert t.sustained_seconds() >= 0
    _patch_psutil(monkeypatch, free_mb=8000, load=0.5)
    t.sample()
    assert t.sustained_seconds() == 0.0
