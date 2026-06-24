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


def test_post_error_does_not_leak_inline_secret(client):
    """A malformed record carrying an inline secret must not echo it in the error."""
    c, cfg_path = client
    _write(cfg_path, {"llm_provider_registry": []})
    r = c.post("/providers", json={"registry": [
        {"name": "x", "level": "BAD", "kind": "openai_compatible",
         "api_key": {"source": "inline", "ref": "sk-LEAK-ME"}}]})
    body = r.json()
    assert body["ok"] is False
    assert "sk-LEAK-ME" not in json.dumps(body)


def test_usage_panel_folds_metering_onto_provider(client, monkeypatch):
    """The Usage column reflects llm_call metering; a gateway model metered with
    its ``openai/`` route prefix folds onto the registry provider that owns it."""
    c, cfg_path = client
    _write(cfg_path, {"llm_provider_registry": [
        {"name": "gw", "level": "L3", "kind": "openai_compatible", "api_base": "https://gw/v1",
         "api_key": {"source": "inline", "ref": "sk"}, "enabled": True, "order": 30,
         "models": [{"id": "zai-org/glm-4.7-maas", "complexity": "light"}]}]})

    class _FakeStore:
        def get_llm_stats(self, limit=5000):
            return {"by_model": {
                # metered under the openai/ dispatch prefix — must still match.
                "openai/zai-org/glm-4.7-maas": {
                    "count": 7, "errors": 2, "total_tokens": 1234, "duration_ms": 100},
            }}

    import skill_hub.store as store_mod
    monkeypatch.setattr(store_mod, "get_store", lambda: _FakeStore())

    r = c.get("/providers")
    assert r.status_code == 200
    assert "5 ok" in r.text       # 7 calls - 2 errors
    assert "2 err" in r.text
    assert "1234 tok" in r.text


def test_post_strips_usage_view_field(client):
    """Saving the page view (which carries a metering ``usage`` block) must not
    persist that field into the registry."""
    c, cfg_path = client
    _write(cfg_path, {"llm_provider_registry": []})
    view = {"registry": [
        {"name": "a", "level": "L1", "kind": "ollama", "api_base": "", "api_key": {},
         "enabled": True, "order": 10, "models": [],
         "usage": {"calls": 9, "ok": 9, "errors": 0, "tokens": 50}}]}
    r = c.post("/providers", json=view)
    assert r.json()["ok"] is True
    assert "usage" not in cfg_mod.get("llm_provider_registry")[0]


def test_post_from_view_preserves_stored_credential(client):
    """Saving the secret-free page view (no api_key) must not wipe the stored cred."""
    c, cfg_path = client
    _write(cfg_path, {"llm_provider_registry": [
        {"name": "gw", "level": "L3", "kind": "openai_compatible", "api_base": "https://gw/v1",
         "api_key": {"source": "inline", "ref": "sk-KEEP"}, "enabled": True, "order": 30,
         "models": [{"id": "m1", "complexity": "light"}]}]})
    # Post the view shape: cred_label instead of api_key, enabled toggled off.
    view = {"registry": [
        {"name": "gw", "level": "L3", "kind": "openai_compatible", "api_base": "https://gw/v1",
         "cred_label": "inline", "enabled": False, "order": 30,
         "models": [{"id": "m1", "complexity": "light"}]}]}
    r = c.post("/providers", json=view)
    assert r.json()["ok"] is True
    saved = cfg_mod.get("llm_provider_registry")[0]
    assert saved["api_key"] == {"source": "inline", "ref": "sk-KEEP"}  # preserved
    assert saved["enabled"] is False                                   # edit applied
    assert "cred_label" not in saved                                   # view field stripped
