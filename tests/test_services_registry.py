"""Tests for the ServiceRegistry reconciliation logic."""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from skill_hub.services.base import Service, Status  # noqa: E402
from skill_hub.services.registry import ServiceRegistry  # noqa: E402


class FakeService(Service):
    def __init__(self, name: str, initial: Status = "stopped"):
        self.name = name
        self.label = name.title()
        self.description = ""
        self._state: Status = initial
        self.start_calls = 0
        self.stop_calls = 0

    def status(self) -> Status:
        return self._state

    def is_available(self):
        return True, ""

    def start(self):
        self.start_calls += 1
        self._state = "running"
        return True, "ok"

    def stop(self):
        self.stop_calls += 1
        self._state = "stopped"
        return True, "ok"

    def resource_footprint(self):
        return {"ram_mb_approx": 100, "cpu_share": 0.01}


def test_reconcile_starts_enabled_stopped():
    svc = FakeService("alpha")
    reg = ServiceRegistry([svc])
    reg.reconcile({"services": {"alpha": {"enabled": True}}})
    assert svc.start_calls == 1
    assert svc.status() == "running"
    assert "alpha" not in reg._disabled


def test_reconcile_stops_disabled_running():
    svc = FakeService("beta", initial="running")
    reg = ServiceRegistry([svc])
    reg.reconcile({"services": {"beta": {"enabled": False}}})
    assert svc.stop_calls == 1
    assert svc.status() == "stopped"
    assert "beta" in reg._disabled


def test_reconcile_idempotent():
    svc = FakeService("gamma", initial="running")
    reg = ServiceRegistry([svc])
    cfg = {"services": {"gamma": {"enabled": True}}}
    reg.reconcile(cfg)
    reg.reconcile(cfg)
    reg.reconcile(cfg)
    assert svc.start_calls == 0  # already running
    assert svc.stop_calls == 0


def test_disabled_services_label_list():
    alpha = FakeService("alpha", initial="stopped")
    beta = FakeService("beta", initial="running")
    reg = ServiceRegistry([alpha, beta])
    reg.reconcile({"services": {"alpha": {"enabled": False}, "beta": {"enabled": False}}})
    assert reg.disabled_services == ["Alpha", "Beta"]


def test_registry_get_and_all():
    a = FakeService("a")
    b = FakeService("b")
    reg = ServiceRegistry([a, b])
    assert reg.get("a") is a
    assert reg.get("missing") is None
    assert len(reg.all()) == 2
