"""Tests for the /settings/orchestrator endpoint.

Mirrors test_routes_control.py: TestClient + monkeypatched config path so the
real user config is never touched.
"""
from __future__ import annotations

import json
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
    """Return a TestClient with config pointing at a fresh tmp dir."""
    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", tmp_path / "config.json")

    # Minimal service registry so create_app doesn't crash.
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
    return TestClient(app), tmp_path / "config.json"


# ---------------------------------------------------------------------------
# GET /settings — orchestrator panel renders
# ---------------------------------------------------------------------------

class TestSettingsPageRenders:
    def test_page_includes_orchestrator_section(self, client):
        c, _ = client
        r = c.get("/settings")
        assert r.status_code == 200
        assert "Tooling Orchestrator" in r.text

    def test_page_includes_all_four_modes(self, client):
        c, _ = client
        r = c.get("/settings")
        for mode in ("off", "offer", "auto", "everywhere"):
            assert mode in r.text, f"mode {mode!r} not found in page"

    def test_page_includes_parent_folders_editor(self, client):
        c, _ = client
        r = c.get("/settings")
        assert "orchestrator_auto_init_roots" in r.text
        assert "Parent folders" in r.text


# ---------------------------------------------------------------------------
# POST /settings/orchestrator — valid mode persists
# ---------------------------------------------------------------------------

class TestOrchestratorSaveValid:
    def test_save_off_mode(self, client):
        c, cfg_path = client
        r = c.post("/settings/orchestrator", data={"orchestrator_mode": "off",
                                                    "orchestrator_auto_init_roots": ""})
        assert r.status_code == 200
        assert "ok" in r.text
        data = json.loads(cfg_path.read_text())
        assert data["orchestrator_mode"] == "off"

    def test_save_offer_mode(self, client):
        c, cfg_path = client
        r = c.post("/settings/orchestrator", data={"orchestrator_mode": "offer",
                                                    "orchestrator_auto_init_roots": ""})
        assert r.status_code == 200
        data = json.loads(cfg_path.read_text())
        assert data["orchestrator_mode"] == "offer"

    def test_save_auto_mode(self, client):
        c, cfg_path = client
        r = c.post("/settings/orchestrator", data={"orchestrator_mode": "auto",
                                                    "orchestrator_auto_init_roots": "/tmp/projects"})
        assert r.status_code == 200
        data = json.loads(cfg_path.read_text())
        assert data["orchestrator_mode"] == "auto"

    def test_save_everywhere_mode(self, client):
        c, cfg_path = client
        r = c.post("/settings/orchestrator", data={"orchestrator_mode": "everywhere",
                                                    "orchestrator_auto_init_roots": ""})
        assert r.status_code == 200
        data = json.loads(cfg_path.read_text())
        assert data["orchestrator_mode"] == "everywhere"

    def test_response_is_html_span(self, client):
        c, _ = client
        r = c.post("/settings/orchestrator", data={"orchestrator_mode": "offer",
                                                    "orchestrator_auto_init_roots": ""})
        assert "<span" in r.text
        assert "status" in r.text


# ---------------------------------------------------------------------------
# POST /settings/orchestrator — invalid mode is rejected
# ---------------------------------------------------------------------------

class TestOrchestratorSaveInvalid:
    def test_invalid_mode_returns_error_span(self, client):
        c, cfg_path = client
        r = c.post("/settings/orchestrator", data={"orchestrator_mode": "banana",
                                                    "orchestrator_auto_init_roots": ""})
        assert r.status_code == 200
        assert "err" in r.text
        assert "Invalid mode" in r.text

    def test_invalid_mode_does_not_write_config(self, client):
        c, cfg_path = client
        r = c.post("/settings/orchestrator", data={"orchestrator_mode": "turbo",
                                                    "orchestrator_auto_init_roots": ""})
        # Config file should either not exist or not have the invalid value.
        if cfg_path.exists():
            data = json.loads(cfg_path.read_text())
            assert data.get("orchestrator_mode") != "turbo"

    def test_empty_mode_is_rejected(self, client):
        c, _ = client
        r = c.post("/settings/orchestrator", data={"orchestrator_mode": "",
                                                    "orchestrator_auto_init_roots": ""})
        assert "err" in r.text

    def test_invalid_mode_is_html_escaped(self, client):
        # The rejected value is reflected into an HTMLResponse the client swaps
        # into the DOM — it must be escaped, never echoed as live markup.
        c, _ = client
        payload = "<script>alert(1)</script>"
        r = c.post("/settings/orchestrator", data={
            "orchestrator_mode": payload,
            "orchestrator_auto_init_roots": "",
        })
        assert "<script>alert(1)</script>" not in r.text
        assert "&lt;script&gt;" in r.text


# ---------------------------------------------------------------------------
# POST /settings/orchestrator — folder list parsing
# ---------------------------------------------------------------------------

class TestFolderListParsing:
    def test_multi_line_roots_parsed(self, client):
        c, cfg_path = client
        r = c.post(
            "/settings/orchestrator",
            data={
                "orchestrator_mode": "auto",
                "orchestrator_auto_init_roots": "/work/code\n/home/user/projects\n",
            },
        )
        assert r.status_code == 200
        data = json.loads(cfg_path.read_text())
        roots = data["orchestrator_auto_init_roots"]
        assert "/work/code" in roots
        assert "/home/user/projects" in roots

    def test_blank_lines_dropped(self, client):
        c, cfg_path = client
        r = c.post(
            "/settings/orchestrator",
            data={
                "orchestrator_mode": "auto",
                "orchestrator_auto_init_roots": "\n/work/code\n\n  \n",
            },
        )
        data = json.loads(cfg_path.read_text())
        roots = data["orchestrator_auto_init_roots"]
        assert "" not in roots
        assert "  " not in roots
        assert "/work/code" in roots

    def test_empty_roots_stores_empty_list(self, client):
        c, cfg_path = client
        c.post("/settings/orchestrator", data={"orchestrator_mode": "auto",
                                               "orchestrator_auto_init_roots": ""})
        data = json.loads(cfg_path.read_text())
        assert data["orchestrator_auto_init_roots"] == []

    def test_roots_stored_for_non_auto_mode(self, client):
        """Roots should be stored regardless of mode — just not used unless mode=auto."""
        c, cfg_path = client
        c.post(
            "/settings/orchestrator",
            data={
                "orchestrator_mode": "everywhere",
                "orchestrator_auto_init_roots": "/my/path",
            },
        )
        data = json.loads(cfg_path.read_text())
        assert "/my/path" in data["orchestrator_auto_init_roots"]


# ---------------------------------------------------------------------------
# Round-trip: save then reload shows new mode
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_saved_mode_shows_on_reload(self, client):
        c, _ = client
        # Save "auto" mode.
        c.post("/settings/orchestrator", data={"orchestrator_mode": "auto",
                                               "orchestrator_auto_init_roots": "/tmp"})
        # Reload settings page — should reflect saved mode.
        r = c.get("/settings")
        assert r.status_code == 200
        # The effective mode badge or the selected radio card should say "auto".
        assert "auto" in r.text

    def test_saved_roots_show_on_reload(self, client):
        c, _ = client
        c.post(
            "/settings/orchestrator",
            data={
                "orchestrator_mode": "auto",
                "orchestrator_auto_init_roots": "/my/special/root",
            },
        )
        r = c.get("/settings")
        assert r.status_code == 200
        assert "/my/special/root" in r.text
