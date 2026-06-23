"""Tests for GET /control/hooks/status and POST /control/hooks/restore."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import pytest
from fastapi.testclient import TestClient

from skill_hub import base_config as bc
from skill_hub import config as cfg_mod
from skill_hub.services import registry as reg_mod


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

class _FakePressure:
    def sample(self):
        from skill_hub.services.monitor import ResourceSample
        return ResourceSample(8000, 16000, 1.0, 8, False, 0.0)

    def sustained_seconds(self):
        return 0.0

    def last_sample(self):
        return self.sample()


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Redirect config so no real user files are touched.
    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", tmp_path / "config.json")

    # Redirect all three base-config paths via env overrides.
    settings_p = tmp_path / "settings.json"
    claude_json_p = tmp_path / "claude.json"
    claude_md_p = tmp_path / "CLAUDE.md"
    monkeypatch.setenv("SKILL_HUB_CLAUDE_JSON", str(claude_json_p))
    monkeypatch.setenv("SKILL_HUB_CLAUDE_MD", str(claude_md_p))
    # Re-read the module-level path constants so they pick up env overrides.
    monkeypatch.setattr(bc, "SETTINGS_PATH", settings_p)
    monkeypatch.setattr(bc, "CLAUDE_JSON_PATH", claude_json_p)
    monkeypatch.setattr(bc, "CLAUDE_MD_PATH", claude_md_p)

    reg = reg_mod.ServiceRegistry([])
    reg_mod.set_registry(reg)
    reg_mod.set_pressure(_FakePressure())

    from skill_hub.webapp.main import create_app
    app = create_app(store=None)
    return TestClient(app), tmp_path


# ---------------------------------------------------------------------------
# GET /control/hooks/status
# ---------------------------------------------------------------------------

class TestHooksStatusRoute:
    def test_returns_200(self, client):
        c, _ = client
        r = c.get("/control/hooks/status")
        assert r.status_code == 200

    def test_shows_three_sections(self, client):
        c, _ = client
        r = c.get("/control/hooks/status")
        assert "Hook Registration" in r.text
        assert "MCP Server Registration" in r.text
        assert "Base Roles Block" in r.text

    def test_shows_missing_items_when_blank(self, client):
        c, _ = client
        r = c.get("/control/hooks/status")
        # No files exist yet — all three sections should report issues.
        assert r.status_code == 200
        # At least one section should not show "All hooks present"
        assert "missing" in r.text.lower() or "not found" in r.text.lower() or "not present" in r.text.lower()

    def test_shows_all_present_after_restore(self, client):
        c, tmp = client
        # Pre-install everything.
        settings_p = tmp / "settings.json"
        claude_json_p = tmp / "claude.json"
        claude_md_p = tmp / "CLAUDE.md"
        bc.install(settings_p, backup=False)
        bc.install_mcp(claude_json_p, backup=False)
        bc.install_roles(claude_md_p, backup=False)

        r = c.get("/control/hooks/status")
        assert r.status_code == 200
        assert "All hooks present" in r.text
        assert "skill-hub registered" in r.text
        assert "Base-roles block present" in r.text

    def test_restore_button_present(self, client):
        c, _ = client
        r = c.get("/control/hooks/status")
        assert 'hx-post="/control/hooks/restore"' in r.text


# ---------------------------------------------------------------------------
# POST /control/hooks/restore
# ---------------------------------------------------------------------------

class TestHooksRestoreRoute:
    def test_returns_200(self, client):
        c, _ = client
        r = c.post("/control/hooks/restore")
        assert r.status_code == 200

    def test_restore_writes_files(self, client):
        c, tmp = client
        settings_p = tmp / "settings.json"
        claude_json_p = tmp / "claude.json"
        claude_md_p = tmp / "CLAUDE.md"
        assert not settings_p.exists()

        c.post("/control/hooks/restore")

        assert settings_p.exists()
        assert claude_json_p.exists()
        assert claude_md_p.exists()

    def test_restore_shows_summary(self, client):
        c, _ = client
        r = c.post("/control/hooks/restore")
        assert "Restored" in r.text or "Nothing to restore" in r.text

    def test_restore_idempotent_second_call(self, client):
        c, _ = client
        c.post("/control/hooks/restore")
        r2 = c.post("/control/hooks/restore")
        assert r2.status_code == 200
        assert "Nothing to restore" in r2.text

    def test_response_contains_status_sections(self, client):
        c, _ = client
        r = c.post("/control/hooks/restore")
        assert "Hook Registration" in r.text
        assert "MCP Server Registration" in r.text
        assert "Base Roles Block" in r.text


# ---------------------------------------------------------------------------
# Control page contains Hooks tab
# ---------------------------------------------------------------------------

class TestHooksTabInControlPage:
    def test_control_page_has_hooks_tab_button(self, client):
        c, _ = client
        r = c.get("/control")
        assert r.status_code == 200
        assert "Hooks" in r.text

    def test_control_page_has_hooks_htmx_lazy_load(self, client):
        c, _ = client
        r = c.get("/control")
        assert '/control/hooks/status' in r.text
