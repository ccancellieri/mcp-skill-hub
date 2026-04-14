"""Tests for the /control webapp routes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from skill_hub import config as cfg_mod  # noqa: E402
from skill_hub.services import registry as reg_mod  # noqa: E402
from skill_hub.services.base import Service, Status  # noqa: E402


class _FakeService(Service):
    def __init__(self, name: str):
        self.name = name
        self.label = name.title()
        self.description = f"fake {name}"
        self._state: Status = "stopped"

    def status(self) -> Status:
        return self._state

    def is_available(self):
        return True, ""

    def start(self):
        self._state = "running"
        return True, "ok"

    def stop(self):
        self._state = "stopped"
        return True, "ok"


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Point config at a tmp dir so tests don't touch the real user config.
    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", tmp_path / "config.json")

    # Install fake registry.
    svc = _FakeService("alpha")
    reg = reg_mod.ServiceRegistry([svc])
    reg_mod.set_registry(reg)

    # Stub pressure so /control/monitor works.
    class _Pressure:
        def sample(self):
            from skill_hub.services.monitor import ResourceSample
            return ResourceSample(8000, 16000, 1.0, 8, False, 0.0)

        def sustained_seconds(self):
            return 0.0

        def last_sample(self):
            return self.sample()

    reg_mod.set_pressure(_Pressure())

    from skill_hub.webapp.main import create_app

    app = create_app(store=None)
    return TestClient(app), svc, tmp_path / "config.json"


def test_control_page_renders(client):
    c, _, _ = client
    r = c.get("/control")
    assert r.status_code == 200
    assert "Control Panel" in r.text
    assert "Alpha" in r.text  # service label


def test_start_persists_enabled_true(client):
    c, svc, cfg_path = client
    r = c.post("/control/alpha/start")
    assert r.status_code == 200
    assert svc.status() == "running"
    data = json.loads(cfg_path.read_text())
    assert data["services"]["alpha"]["enabled"] is True


def test_stop_persists_enabled_false(client):
    c, svc, cfg_path = client
    svc._state = "running"
    r = c.post("/control/alpha/stop")
    assert r.status_code == 200
    assert svc.status() == "stopped"
    data = json.loads(cfg_path.read_text())
    assert data["services"]["alpha"]["enabled"] is False


def test_toggle_flips_state(client):
    c, svc, cfg_path = client
    assert svc.status() == "stopped"
    c.post("/control/alpha/toggle")
    assert svc.status() == "running"
    c.post("/control/alpha/toggle")
    assert svc.status() == "stopped"


def test_card_endpoint_returns_partial(client):
    c, _, _ = client
    r = c.get("/control/alpha/card")
    assert r.status_code == 200
    assert "status-dot" in r.text
    assert "Alpha" in r.text


def test_monitor_endpoint_returns_partial(client):
    c, _, _ = client
    r = c.get("/control/monitor")
    assert r.status_code == 200
    assert "monitor-bar" in r.text


def test_unknown_service_404(client):
    c, _, _ = client
    assert c.post("/control/missing/start").status_code == 404


def test_banner_appears_when_service_disabled(client):
    c, svc, _ = client
    # Reconcile to populate disabled_services on the registry.
    reg_mod.get_registry().reconcile({"services": {"alpha": {"enabled": False}}})
    r = c.get("/control")
    assert r.status_code == 200
    assert "⚠ Disabled: Alpha" in r.text
