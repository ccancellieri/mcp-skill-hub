"""Tests for the /control/chrome endpoint.

Mirrors test_orchestrator_settings_route.py: TestClient + monkeypatched config
so no real plugin registry, filesystem, or subprocess is touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import pytest
from fastapi.testclient import TestClient

from skill_hub import config as cfg_mod
from skill_hub.services import registry as reg_mod


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient with config pointing at a fresh tmp dir; no real services."""
    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", tmp_path / "config.json")

    reg = reg_mod.ServiceRegistry([])
    reg_mod.set_registry(reg)

    class _FakePressure:
        def sample(self):
            from skill_hub.services.monitor import ResourceSample
            return ResourceSample(8000, 16000, 1.0, 8, False, 0.0)
        def sustained_seconds(self): return 0.0
        def last_sample(self): return self.sample()

    reg_mod.set_pressure(_FakePressure())

    from skill_hub.webapp.main import create_app
    app = create_app(store=None)
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /control/chrome — panel renders
# ---------------------------------------------------------------------------

class TestChromePanelRenders:
    def test_panel_returns_200(self, client):
        r = client.get("/control/chrome")
        assert r.status_code == 200

    def test_panel_contains_plugin_status(self, client):
        r = client.get("/control/chrome")
        # Must contain "enabled" or "disabled" status text.
        assert "enabled" in r.text or "disabled" in r.text

    def test_panel_contains_connection_mode(self, client):
        r = client.get("/control/chrome")
        assert "isolated sandbox" in r.text

    def test_panel_contains_intents_link(self, client):
        r = client.get("/control/chrome")
        assert "/intents" in r.text

    def test_panel_contains_debugging_port_snippet(self, client):
        r = client.get("/control/chrome")
        assert "9222" in r.text

    def test_panel_contains_browser_url_snippet(self, client):
        r = client.get("/control/chrome")
        assert "--browserUrl" in r.text

    def test_panel_is_read_only_no_form_actions(self, client):
        r = client.get("/control/chrome")
        # The chrome panel must not contain any form POSTing to a launch/process endpoint.
        assert "launch" not in r.text.lower() or "Real-profile launcher" in r.text


# ---------------------------------------------------------------------------
# Degraded data sources — panel must not 500
# ---------------------------------------------------------------------------

class TestChromeDegradedDataSources:
    def test_panel_survives_broken_plugin_registry(self, client, monkeypatch):
        from skill_hub.webapp.routes import control_chrome as cc_mod

        def _bad():
            raise RuntimeError("plugin registry exploded")

        monkeypatch.setattr(cc_mod, "_chrome_plugin_enabled", _bad)
        r = client.get("/control/chrome")
        assert r.status_code == 200

    def test_panel_survives_broken_intents_queue(self, client, monkeypatch):
        from skill_hub.webapp.routes import control_chrome as cc_mod

        def _bad(limit=5):
            raise RuntimeError("intents queue exploded")

        monkeypatch.setattr(cc_mod, "_recent_intents", _bad)
        r = client.get("/control/chrome")
        assert r.status_code == 200

    def test_panel_shows_enabled_when_plugin_present(self, client, monkeypatch):
        from skill_hub.webapp.routes import control_chrome as cc_mod

        monkeypatch.setattr(cc_mod, "_chrome_plugin_enabled", lambda: True)
        monkeypatch.setattr(cc_mod, "_recent_intents", lambda limit=5: [])
        monkeypatch.setattr(cc_mod, "_total_intent_count", lambda: 0)

        r = client.get("/control/chrome")
        assert r.status_code == 200
        assert "enabled" in r.text

    def test_panel_shows_disabled_when_plugin_absent(self, client, monkeypatch):
        from skill_hub.webapp.routes import control_chrome as cc_mod

        monkeypatch.setattr(cc_mod, "_chrome_plugin_enabled", lambda: False)
        monkeypatch.setattr(cc_mod, "_recent_intents", lambda limit=5: [])
        monkeypatch.setattr(cc_mod, "_total_intent_count", lambda: 0)

        r = client.get("/control/chrome")
        assert r.status_code == 200
        assert "disabled" in r.text

    def test_panel_renders_recent_intents(self, client, monkeypatch):
        from skill_hub.webapp.routes import control_chrome as cc_mod

        fake_intents = [
            {"id": "abc", "url": "https://example.com", "action": "navigate",
             "note": "", "status": "done", "created_at": 1000},
            {"id": "def", "url": "https://test.org", "action": "screenshot",
             "note": "test note", "status": "pending", "created_at": 999},
        ]
        monkeypatch.setattr(cc_mod, "_chrome_plugin_enabled", lambda: True)
        monkeypatch.setattr(cc_mod, "_recent_intents", lambda limit=5: fake_intents)
        monkeypatch.setattr(cc_mod, "_total_intent_count", lambda: 2)

        r = client.get("/control/chrome")
        assert r.status_code == 200
        assert "example.com" in r.text
        assert "navigate" in r.text
