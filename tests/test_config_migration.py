"""Tests for legacy-key migration into the services dict."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from skill_hub import config as cfg_mod  # noqa: E402


def _point_config_at(monkeypatch, tmp_path: Path) -> Path:
    path = tmp_path / "config.json"
    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", path)
    return path


def test_legacy_searxng_enabled_folded(tmp_path, monkeypatch):
    path = _point_config_at(monkeypatch, tmp_path)
    path.write_text(json.dumps({"searxng_enabled": False}))

    loaded = cfg_mod.load_config()
    assert loaded["services"]["searxng"]["enabled"] is False
    assert "searxng_enabled" not in loaded

    # Disk was rewritten with the migrated shape.
    on_disk = json.loads(path.read_text())
    assert on_disk["services"]["searxng"]["enabled"] is False
    assert "searxng_enabled" not in on_disk


def test_legacy_router_haiku_folded(tmp_path, monkeypatch):
    path = _point_config_at(monkeypatch, tmp_path)
    path.write_text(json.dumps({"router_haiku_enabled": True}))

    loaded = cfg_mod.load_config()
    assert loaded["services"]["haiku_router"]["enabled"] is True
    assert "router_haiku_enabled" not in loaded


def test_legacy_router_ollama_model_folded(tmp_path, monkeypatch):
    path = _point_config_at(monkeypatch, tmp_path)
    path.write_text(json.dumps({"router_ollama_model": "llama3:8b"}))

    loaded = cfg_mod.load_config()
    assert loaded["services"]["ollama_router"]["model"] == "llama3:8b"
    assert "router_ollama_model" not in loaded


def test_migration_idempotent(tmp_path, monkeypatch):
    path = _point_config_at(monkeypatch, tmp_path)
    path.write_text(json.dumps({"searxng_enabled": False, "router_haiku_enabled": True}))

    first = cfg_mod.load_config()
    second = cfg_mod.load_config()
    assert first["services"]["searxng"]["enabled"] is False
    assert second["services"]["searxng"]["enabled"] is False
    assert second["services"]["haiku_router"]["enabled"] is True


def test_new_config_untouched(tmp_path, monkeypatch):
    path = _point_config_at(monkeypatch, tmp_path)
    path.write_text(json.dumps({"services": {"searxng": {"enabled": False}}}))

    loaded = cfg_mod.load_config()
    assert loaded["services"]["searxng"]["enabled"] is False
    # Defaults filled in for missing services.
    assert "ollama_router" in loaded["services"]
    assert loaded["services"]["ollama_router"]["model"] == "qwen2.5:3b"


def test_service_helpers(tmp_path, monkeypatch):
    _point_config_at(monkeypatch, tmp_path)
    # No file → defaults.
    assert cfg_mod.is_service_enabled("searxng") is True
    assert cfg_mod.service_field("ollama_router", "model") == "qwen2.5:3b"
    assert cfg_mod.is_service_enabled("haiku_router") is False


def test_respects_existing_service_value_over_legacy(tmp_path, monkeypatch):
    path = _point_config_at(monkeypatch, tmp_path)
    path.write_text(json.dumps({
        "searxng_enabled": True,  # legacy True
        "services": {"searxng": {"enabled": False}},  # explicit new wins
    }))
    loaded = cfg_mod.load_config()
    assert loaded["services"]["searxng"]["enabled"] is False
    assert "searxng_enabled" not in loaded
