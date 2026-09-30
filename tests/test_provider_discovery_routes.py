from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skill_hub import config as _config  # noqa: E402
from skill_hub.webapp.routes import providers as prov_routes  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "config.json")
    app = FastAPI()
    app.include_router(prov_routes.router)
    return TestClient(app)


def _seed(enabled=True, kind="openai_compatible", api_key=None):
    _config.set("llm_provider_registry", [{
        "name": "gateway", "kind": kind, "api_base": "https://gateway.example/v1",
        "api_key": api_key or {"source": "inline", "ref": "secret-token"},
        "enabled": enabled, "order": 30,
        "models": [{"id": "existing-model", "complexity": "heavy", "tags": ["code"]}],
    }])


def test_discover_uses_resolved_credentials_and_returns_read_only_proposals(client, monkeypatch):
    _seed(enabled=False)
    observed = {}

    class FakeClient:
        def __init__(self, **kwargs):
            observed["kwargs"] = kwargs
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, method, url, **kwargs):
            observed.update(url=url, get_kwargs=kwargs)
            return FakeResponse(httpx.Response(200, json={"data": [
                {"id": "existing-model"}, {"id": "org/new-model"},
                {"id": "org/new-model"}, {"id": "<img src=x>"}, {"id": ""},
            ]}))

    class FakeResponse:
        def __init__(self, response): self.response = response
        async def __aenter__(self): return self.response
        async def __aexit__(self, *args): pass

    monkeypatch.setattr(prov_routes.httpx, "AsyncClient", FakeClient)
    before = _config.get("llm_provider_registry")
    response = client.post("/providers/discover", json={"name": "gateway"})

    assert response.status_code == 200
    body = response.json()
    assert body["configured"] == ["existing-model"]
    assert body["proposed"] == [{
        "id": "org/new-model", "qualified_id": "gateway::org/new-model",
        "availability": "unknown", "cost": "unknown",
    }, {
        "id": "<img src=x>", "qualified_id": "gateway::<img src=x>",
        "availability": "unknown", "cost": "unknown",
    }]
    assert observed["url"] == "https://gateway.example/v1/models"
    assert observed["get_kwargs"]["headers"] == {"Authorization": "Bearer secret-token"}
    assert observed["get_kwargs"]["follow_redirects"] is False
    assert observed["kwargs"]["timeout"] <= 5
    assert _config.get("llm_provider_registry") == before


@pytest.mark.parametrize("upstream", [
    httpx.Response(503, text="secret-token upstream diagnostic"),
    httpx.Response(200, text="not-json"),
    httpx.Response(200, json={"data": [{"id": "bad\nid"}]}),
])
def test_discover_sanitizes_upstream_errors_and_invalid_models(client, monkeypatch, upstream):
    _seed()

    class FakeClient:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs):
            class Ctx:
                async def __aenter__(self): return upstream
                async def __aexit__(self, *args): pass
            return Ctx()

    monkeypatch.setattr(prov_routes.httpx, "AsyncClient", FakeClient)
    response = client.post("/providers/discover", json={"name": "gateway"})
    assert response.status_code == 200
    assert "secret-token" not in response.text
    if upstream.status_code >= 400 or upstream.text == "not-json":
        assert response.json()["ok"] is False
    else:
        assert response.json()["proposed"] == []


def test_discover_handles_timeout_without_leaking_exception(client, monkeypatch):
    _seed()

    class FakeClient:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs):
            class Ctx:
                async def __aenter__(self): raise httpx.ReadTimeout("secret-token")
                async def __aexit__(self, *args): pass
            return Ctx()

    monkeypatch.setattr(prov_routes.httpx, "AsyncClient", FakeClient)
    response = client.post("/providers/discover", json={"name": "gateway"})
    assert response.status_code == 200
    assert response.json() == {"ok": False, "error": "Provider discovery failed"}
    assert "secret-token" not in response.text


def test_discover_rejects_unknown_unsupported_and_unresolved_providers(client, monkeypatch):
    _seed(kind="ollama")
    assert client.post("/providers/discover", json={"name": "missing"}).json()["ok"] is False
    assert client.post("/providers/discover", json={"name": "gateway"}).json()["ok"] is False
    _seed(api_key={"source": "env", "ref": "ABSENT_DISCOVERY_KEY"})
    assert client.post("/providers/discover", json={"name": "gateway"}).json()["ok"] is False


def test_discover_rejects_invalid_provider_name(client):
    response = client.post("/providers/discover", json={"name": "../gateway"})
    assert response.status_code == 200
    assert response.json()["ok"] is False


def test_discover_reports_catalog_truncation(client, monkeypatch):
    _seed()
    catalog = httpx.Response(200, json={"data": [{"id": f"model-{n}"} for n in range(1001)]})

    class FakeClient:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs):
            class Ctx:
                async def __aenter__(self): return catalog
                async def __aexit__(self, *args): pass
            return Ctx()

    monkeypatch.setattr(prov_routes.httpx, "AsyncClient", FakeClient)
    body = client.post("/providers/discover", json={"name": "gateway"}).json()
    assert len(body["proposed"]) == 1000
    assert body["truncated"] is True


def test_discover_rejects_redirect_response_even_with_catalog_json(client, monkeypatch):
    _seed()
    upstream = httpx.Response(302, json={"data": [{"id": "redirect-model"}]},
                              headers={"location": "https://other.example/models"})

    class FakeClient:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs):
            class Ctx:
                async def __aenter__(self): return upstream
                async def __aexit__(self, *args): pass
            return Ctx()

    monkeypatch.setattr(prov_routes.httpx, "AsyncClient", FakeClient)
    assert client.post("/providers/discover", json={"name": "gateway"}).json() == {
        "ok": False, "error": "Provider discovery failed",
    }


def test_discover_rejects_ids_with_edge_whitespace(client, monkeypatch):
    _seed()
    upstream = httpx.Response(200, json={"data": [
        {"id": " leading"}, {"id": "trailing "}, {"id": "valid/id"},
    ]})

    class FakeClient:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs):
            class Ctx:
                async def __aenter__(self): return upstream
                async def __aexit__(self, *args): pass
            return Ctx()

    monkeypatch.setattr(prov_routes.httpx, "AsyncClient", FakeClient)
    body = client.post("/providers/discover", json={"name": "gateway"}).json()
    assert [row["id"] for row in body["proposed"]] == ["valid/id"]


def test_discover_deadline_covers_entire_stream(client, monkeypatch):
    _seed()
    monkeypatch.setattr(prov_routes, "_DISCOVERY_TIMEOUT_SECONDS", 0.05)

    class SlowResponse:
        status_code = 200

        async def aiter_bytes(self):
            yield b'{"data": ['
            await asyncio.sleep(0.08)
            yield b'{"id": "late-model"}]}'

    class FakeClient:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs):
            class Ctx:
                async def __aenter__(self): return SlowResponse()
                async def __aexit__(self, *args): pass
            return Ctx()

    monkeypatch.setattr(prov_routes.httpx, "AsyncClient", FakeClient)
    assert client.post("/providers/discover", json={"name": "gateway"}).json() == {
        "ok": False, "error": "Provider discovery failed",
    }
