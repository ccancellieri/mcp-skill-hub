"""The manual composer can start without starting configured services."""
from __future__ import annotations

import sys

from fastapi.testclient import TestClient


def test_dashboard_no_services_keeps_composer_available(tmp_path, monkeypatch):
    from skill_hub.webapp import __main__ as launcher
    from skill_hub.services import registry
    from skill_hub import cron, system_health

    config = {"services": {"auto_reconcile": True}}
    monkeypatch.setattr(launcher, "_load_cfg", lambda: config)
    monkeypatch.setattr(launcher, "DB", tmp_path / "dashboard.db")
    monkeypatch.setattr(sys, "argv", ["skill-hub-dashboard", "--no-services"])

    attempted = []

    def unexpected(*args, **kwargs):
        attempted.append(True)
        raise AssertionError("service startup was attempted")

    monkeypatch.setattr(registry.ServiceRegistry, "build_from_config", unexpected)
    monkeypatch.setattr(registry, "start_reconciler", unexpected)
    monkeypatch.setattr(cron, "seed_defaults", unexpected)
    monkeypatch.setattr(system_health, "start_health_watcher", unexpected)
    served = []

    def serve(app, **kwargs):
        with TestClient(app) as client:
            response = client.get("/context")
            assert response.status_code == 200
            assert 'id="composer-prompt"' in response.text
        served.append(kwargs)

    monkeypatch.setattr(launcher.uvicorn, "run", serve)
    assert launcher.main() == 0
    assert attempted == []
    assert served[0]["host"] == "127.0.0.1"
    assert config == {"services": {"auto_reconcile": True}}


def test_dashboard_default_retains_service_reconciliation(tmp_path, monkeypatch):
    from skill_hub.webapp import __main__ as launcher
    from skill_hub.services import registry

    monkeypatch.setattr(launcher, "_load_cfg", lambda: {})
    monkeypatch.setattr(launcher, "DB", tmp_path / "dashboard.db")
    monkeypatch.setattr(sys, "argv", ["skill-hub-dashboard"])
    started = []

    class Handle:
        def stop(self):
            pass

    def start(services, pressure, config_path, load_config, **kwargs):
        assert isinstance(services, registry.ServiceRegistry)
        started.append(kwargs)
        return Handle()

    monkeypatch.setattr(registry, "start_reconciler", start)
    monkeypatch.setattr(registry, "set_registry", lambda value: None)
    monkeypatch.setattr(launcher.uvicorn, "run", lambda *args, **kwargs: None)
    assert launcher.main() == 0
    assert len(started) == 1
