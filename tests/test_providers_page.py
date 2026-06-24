"""Tests for the /providers settings page."""
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
    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", tmp_path / "config.json")
    reg_mod.set_registry(reg_mod.ServiceRegistry([]))

    class _FakePressure:
        def sample(self):
            from skill_hub.services.monitor import ResourceSample
            return ResourceSample(8000, 16000, 1.0, 8, False, 0.0)
        def sustained_seconds(self): return 0.0
        def last_sample(self): return self.sample()
    reg_mod.set_pressure(_FakePressure())

    from skill_hub.webapp.main import create_app
    return TestClient(create_app(store=None)), cfg_mod.CONFIG_PATH


def _write(path, data):
    path.write_text(json.dumps(data))


def test_providers_page_lists_registry_without_secrets(client):
    c, cfg_path = client
    _write(cfg_path, {"llm_provider_registry": [
        {"name": "gw", "level": "L3", "kind": "openai_compatible", "api_base": "https://gw/v1",
         "api_key": {"source": "inline", "ref": "sk-SECRET"}, "enabled": True, "order": 30,
         "models": [{"id": "m1", "complexity": "light"}]}]})
    r = c.get("/providers")
    assert r.status_code == 200
    assert "gw" in r.text and "sk-SECRET" not in r.text


def test_post_persists_registry(client):
    c, cfg_path = client
    _write(cfg_path, {"llm_provider_registry": []})
    new = {"registry": [
        {"name": "a", "level": "L1", "kind": "ollama", "api_base": "", "api_key": {},
         "enabled": True, "order": 10, "models": []}]}
    r = c.post("/providers", json=new)
    assert r.status_code == 200 and r.json()["ok"] is True
    assert cfg_mod.get("llm_provider_registry")[0]["name"] == "a"


def test_post_rejects_malformed(client):
    c, cfg_path = client
    _write(cfg_path, {"llm_provider_registry": []})
    r = c.post("/providers", json={"registry": [{"name": "x", "level": "BAD", "kind": "ollama"}]})
    assert r.json()["ok"] is False
