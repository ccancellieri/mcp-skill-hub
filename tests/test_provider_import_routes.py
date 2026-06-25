# tests/test_provider_import_routes.py
from __future__ import annotations
import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skill_hub import config as _config        # noqa: E402
from skill_hub.llm import importers as _imp     # noqa: E402
from skill_hub.webapp.routes import providers as prov_routes  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Isolate config writes — point CONFIG_PATH at a tmp file, never the real one.
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    app = FastAPI()
    app.include_router(prov_routes.router)
    return TestClient(app)


OPENCODE = {
    "provider": {
        "agent-platform": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "FAO Vibe Coding",
            "options": {"baseURL": "https://gw/v1", "apiKey": "SECRET"},
            "models": {"m1": {}, "m2": {}},
        }
    }
}


def test_preview_opencode_payload_returns_diff_no_write(client):
    r = client.post("/providers/import/preview",
                    data=json.dumps({"format": "opencode", "payload": OPENCODE}),
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["diff"][0]["status"] == "new"
    assert set(body["diff"][0]["models_added"]) == {"m1", "m2"}
    # preview must not persist anything — the config file is never written
    assert not _config.CONFIG_PATH.exists()


def test_preview_secret_free(client):
    r = client.post("/providers/import/preview",
                    data=json.dumps({"format": "opencode", "payload": OPENCODE}),
                    headers={"Content-Type": "application/json"})
    assert "SECRET" not in r.text
    assert r.json()["diff"][0]["cred_label"] == "opencode:agent-platform"


def test_apply_openai_persists_provider(client):
    payload = {"name": "h", "baseURL": "http://h/v1", "models": ["x"]}
    r = client.post("/providers/import/apply",
                    data=json.dumps({"format": "openai", "payload": payload}),
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 200 and r.json()["ok"] is True
    reg = _config.get("llm_provider_registry")
    assert any(rec["name"] == "h" for rec in reg)


def test_apply_opencode_from_file(client, monkeypatch):
    # No payload → reads the on-disk opencode config (the one-click Sync path).
    monkeypatch.setattr(_imp, "read_opencode_config", lambda: OPENCODE)
    r = client.post("/providers/import/apply",
                    data=json.dumps({"format": "opencode"}),
                    headers={"Content-Type": "application/json"})
    assert r.json()["ok"] is True
    reg = _config.get("llm_provider_registry")
    rec = next(r for r in reg if r["api_key"].get("ref") == "agent-platform")
    assert {m["id"] for m in rec["models"]} == {"m1", "m2"}


def test_unsupported_format_rejected(client):
    r = client.post("/providers/import/preview",
                    data=json.dumps({"format": "bogus", "payload": {}}),
                    headers={"Content-Type": "application/json"})
    assert r.json()["ok"] is False


def test_apply_preserves_existing_tags(client, monkeypatch):
    # Seed an existing record with a tuned model, then sync from opencode.
    _config.set("llm_provider_registry", [{
        "name": "work-gateway", "level": "L3", "kind": "openai_compatible",
        "api_base": "", "api_key": {"source": "opencode", "ref": "agent-platform"},
        "enabled": True, "order": 30,
        "models": [{"id": "m1", "complexity": "heavy", "tags": ["git"]}],
    }])
    monkeypatch.setattr(_imp, "read_opencode_config", lambda: OPENCODE)
    r = client.post("/providers/import/apply",
                    data=json.dumps({"format": "opencode"}),
                    headers={"Content-Type": "application/json"})
    assert r.json()["ok"] is True
    reg = _config.get("llm_provider_registry")
    rec = next(r for r in reg if r["name"] == "work-gateway")
    by_id = {m["id"]: m for m in rec["models"]}
    assert by_id["m1"]["complexity"] == "heavy" and by_id["m1"]["tags"] == ["git"]
    assert "m2" in by_id  # new model added
