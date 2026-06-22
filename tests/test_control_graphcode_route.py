"""Tests for the /control/graphcode endpoints.

Mirrors test_orchestrator_settings_route.py: TestClient + monkeypatched config
so the real user config and real codegraph binary are never touched.
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
# GET /control/graphcode — panel renders
# ---------------------------------------------------------------------------

class TestGraphcodePanelRenders:
    def test_panel_returns_200(self, client):
        r = client.get("/control/graphcode")
        assert r.status_code == 200

    def test_panel_contains_mode_badge(self, client):
        r = client.get("/control/graphcode")
        assert "mode:" in r.text

    def test_panel_contains_probe_form(self, client):
        r = client.get("/control/graphcode")
        assert "graphcode/probe" in r.text

    def test_panel_contains_settings_link(self, client):
        r = client.get("/control/graphcode")
        assert "/settings" in r.text


# ---------------------------------------------------------------------------
# POST /control/graphcode/probe — validate path + html-escape
# ---------------------------------------------------------------------------

class TestGraphcodeProbe:
    def test_probe_nonexistent_path_returns_error(self, client):
        r = client.post("/control/graphcode/probe",
                        data={"path": "/nonexistent/totally/fake/path"})
        assert r.status_code == 200
        assert "err" in r.text

    def test_probe_empty_path_returns_error(self, client):
        r = client.post("/control/graphcode/probe", data={"path": ""})
        assert r.status_code == 200
        assert "err" in r.text

    def test_invalid_path_is_html_escaped(self, client):
        payload = "/tmp/<script>alert(1)</script>"
        r = client.post("/control/graphcode/probe", data={"path": payload})
        assert r.status_code == 200
        assert "<script>alert(1)</script>" not in r.text
        assert "&lt;script&gt;" in r.text

    def test_probe_valid_dir_calls_probe_codegraph(self, client, tmp_path, monkeypatch):
        from skill_hub.orchestrator import engine as eng_mod
        called_with = []

        def _fake_probe(root):
            called_with.append(root)
            from skill_hub.orchestrator.engine import Readiness
            return Readiness(present=False, fresh=False, stale_age=None,
                             detail="not indexed")

        monkeypatch.setattr(eng_mod, "probe_codegraph", _fake_probe)

        r = client.post("/control/graphcode/probe",
                        data={"path": str(tmp_path)})
        assert r.status_code == 200
        assert len(called_with) == 1
        assert called_with[0] == tmp_path


# ---------------------------------------------------------------------------
# POST /control/graphcode/sync — calls ensure_tooling_core
# ---------------------------------------------------------------------------

class TestGraphcodeSync:
    def test_sync_nonexistent_path_returns_error(self, client):
        r = client.post("/control/graphcode/sync",
                        data={"path": "/definitely/not/a/real/path"})
        assert r.status_code == 200
        assert "err" in r.text

    def test_sync_invalid_path_is_html_escaped(self, client):
        payload = "/tmp/<b>injection</b>"
        r = client.post("/control/graphcode/sync", data={"path": payload})
        assert r.status_code == 200
        assert "<b>injection</b>" not in r.text
        assert "&lt;b&gt;" in r.text

    def test_sync_valid_dir_calls_ensure_tooling_core(self, client, tmp_path, monkeypatch):
        from skill_hub.orchestrator import engine as eng_mod
        calls = []

        def _fake_etc(path, *, init=False, refresh=True):
            calls.append({"path": path, "init": init, "refresh": refresh})
            return {"path": path, "present": False, "fresh": False,
                    "action": "none", "directive": ""}

        def _fake_probe(root):
            from skill_hub.orchestrator.engine import Readiness
            return Readiness(present=False, fresh=False, stale_age=None,
                             detail="not indexed")

        monkeypatch.setattr(eng_mod, "ensure_tooling_core", _fake_etc)
        monkeypatch.setattr(eng_mod, "probe_codegraph", _fake_probe)

        r = client.post("/control/graphcode/sync",
                        data={"path": str(tmp_path)})
        assert r.status_code == 200
        assert len(calls) == 1
        assert calls[0]["refresh"] is True
        assert calls[0]["init"] is False


# ---------------------------------------------------------------------------
# POST /control/graphcode/reindex — calls ensure_tooling_core with init=True
# ---------------------------------------------------------------------------

class TestGraphcodeReindex:
    def test_reindex_nonexistent_path_returns_error(self, client):
        r = client.post("/control/graphcode/reindex",
                        data={"path": "/nonexistent/path/xyz"})
        assert r.status_code == 200
        assert "err" in r.text

    def test_reindex_invalid_path_is_html_escaped(self, client):
        payload = '/fake/<img src=x onerror=alert(1)>'
        r = client.post("/control/graphcode/reindex", data={"path": payload})
        assert r.status_code == 200
        assert '<img src=x' not in r.text
        assert '&lt;img' in r.text

    def test_reindex_valid_dir_calls_ensure_tooling_core_with_init(
        self, client, tmp_path, monkeypatch
    ):
        from skill_hub.orchestrator import engine as eng_mod
        calls = []

        def _fake_etc(path, *, init=False, refresh=True):
            calls.append({"path": path, "init": init, "refresh": refresh})
            return {"path": path, "present": False, "fresh": False,
                    "action": "init_run", "directive": ""}

        def _fake_probe(root):
            from skill_hub.orchestrator.engine import Readiness
            return Readiness(present=True, fresh=True, stale_age=5.0,
                             detail="index age 5s (ttl 300s), 42 nodes")

        monkeypatch.setattr(eng_mod, "ensure_tooling_core", _fake_etc)
        monkeypatch.setattr(eng_mod, "probe_codegraph", _fake_probe)

        r = client.post("/control/graphcode/reindex",
                        data={"path": str(tmp_path)})
        assert r.status_code == 200
        assert len(calls) == 1
        assert calls[0]["init"] is True
        assert calls[0]["refresh"] is False


# ---------------------------------------------------------------------------
# Degraded data sources — panel must not 500
# ---------------------------------------------------------------------------

class TestGraphcodeDegradedDataSources:
    def test_panel_survives_broken_resolve_mode(self, client, monkeypatch):
        from skill_hub.webapp.routes import control_graphcode as cg_mod

        def _bad_mode():
            raise RuntimeError("config exploded")

        monkeypatch.setattr(cg_mod, "_resolve_mode", _bad_mode)
        r = client.get("/control/graphcode")
        assert r.status_code == 200

    def test_panel_survives_broken_root_list(self, client, monkeypatch):
        from skill_hub.webapp.routes import control_graphcode as cg_mod

        def _bad_roots(store):
            raise RuntimeError("store exploded")

        monkeypatch.setattr(cg_mod, "_build_root_list", _bad_roots)
        r = client.get("/control/graphcode")
        assert r.status_code == 200
