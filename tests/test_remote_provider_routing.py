"""Tests: remote Ollama (and other endpoints) wired as plain registry rows.

Verifies the key guarantee: a ``kind:"ollama"`` registry row with a non-empty
``api_base`` has that base URL passed through to litellm when the escalation
ladder selects it.  No bespoke scaffold required — the row is just data.

Also verifies that remote embedding is served by the OllamaMultiClient when a
second endpoint is added to ``ollama_endpoints``.
"""
from __future__ import annotations

import json
import importlib

import pytest


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _write_cfg(monkeypatch, tmp_path, data: dict):
    import skill_hub.config as cfg
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data))
    monkeypatch.setattr(cfg, "CONFIG_PATH", p)


def _reload_llm():
    import skill_hub.llm.registry as registry
    import skill_hub.llm.credentials as credentials
    import skill_hub.llm.escalation as escalation
    import skill_hub.llm.provider as provider
    import skill_hub.llm.litellm_adapter as litellm_adapter
    for m in (registry, credentials, escalation, provider, litellm_adapter):
        importlib.reload(m)
    return escalation, litellm_adapter, provider


def _fake_litellm_capture(calls: list[dict]):
    """Return a stub litellm whose completion() records (model, api_base) tuples."""
    class _Stub:
        suppress_debug_info = True
        drop_params = True

        def completion(self, **kwargs):
            calls.append({"model": kwargs.get("model"), "api_base": kwargs.get("api_base")})
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    return _Stub()


# ---------------------------------------------------------------------------
# CHAT: remote-ollama registry row — api_base reaches litellm
# ---------------------------------------------------------------------------

_REMOTE_OLLAMA_REG = {
    "llm_metering_enabled": False,
    "llm_provider_registry": [
        {
            "name": "remote-ollama",
            "level": "L2",
            "kind": "ollama",
            "api_base": "http://ollama.example.internal:11434",
            "api_key": {},
            "enabled": True,
            "order": 20,
            "models": [{"id": "ollama/llama3:8b", "complexity": "light",
                        "tags": ["fast"]}],
        },
    ],
}


def test_remote_ollama_api_base_passed_to_litellm(monkeypatch, tmp_path):
    """The ladder selects the remote-ollama row and passes its api_base to litellm.

    No special code: the registry row IS the remote endpoint — just data.
    """
    _write_cfg(monkeypatch, tmp_path, _REMOTE_OLLAMA_REG)
    escalation, litellm_adapter, provider = _reload_llm()
    escalation.reset_cooldowns()
    # Suppress the reachability probe so the test is network-free.
    monkeypatch.setattr(escalation, "ollama_daemon_reachable", lambda **k: True)

    calls: list[dict] = []
    p = litellm_adapter.LitellmProvider()
    p._litellm = _fake_litellm_capture(calls)

    out = p.chat(
        [provider.Message(role="user", content="hello")],
        complexity=0.1,   # signal → ladder engaged
    )

    assert out == "ok"
    assert len(calls) == 1
    assert calls[0]["model"] == "ollama/llama3:8b"
    assert calls[0]["api_base"] == "http://ollama.example.internal:11434", (
        "remote ollama api_base must be forwarded to litellm.completion"
    )


def test_remote_ollama_env_source_api_base(monkeypatch, tmp_path):
    """api_base can also come from the registry field directly (source omitted).

    resolve_credentials falls through to ``return (provider.api_base or None, None)``
    when no source is set, so the field is used as-is.
    """
    reg = {
        "llm_metering_enabled": False,
        "llm_provider_registry": [
            {
                "name": "remote-ollama-env",
                "level": "L2",
                "kind": "ollama",
                "api_base": "http://ollama.example.internal:11434",
                "api_key": {},  # no source — api_base field is used directly
                "enabled": True,
                "order": 20,
                "models": [{"id": "ollama/mistral:7b", "complexity": "light"}],
            },
        ],
    }
    _write_cfg(monkeypatch, tmp_path, reg)
    escalation, litellm_adapter, provider = _reload_llm()
    escalation.reset_cooldowns()
    monkeypatch.setattr(escalation, "ollama_daemon_reachable", lambda **k: True)

    calls: list[dict] = []
    p = litellm_adapter.LitellmProvider()
    p._litellm = _fake_litellm_capture(calls)

    out = p.chat(
        [provider.Message(role="user", content="hello")],
        complexity=0.1,
    )
    assert out == "ok"
    assert calls[0]["api_base"] == "http://ollama.example.internal:11434"


def test_escalation_select_returns_remote_api_base(monkeypatch):
    """Unit-level: escalation.select() surfaces api_base from a remote-ollama row."""
    from skill_hub.llm import escalation as esc
    from skill_hub.llm.registry import Provider, ProviderModel

    # Inject a fake registry directly — no disk I/O needed.
    remote = Provider(
        name="remote-ollama",
        level="L2",
        kind="ollama",
        api_base="http://ollama.example.internal:11434",
        api_key={},
        enabled=True,
        order=20,
        models=[ProviderModel(id="ollama/llama3:8b", complexity="light")],
    )

    esc.reset_cooldowns()
    # escalation imports load_registry by name into its own namespace; patch there.
    monkeypatch.setattr(esc, "load_registry", lambda: [remote])
    sel = esc.select(0.1)

    assert sel is not None
    assert sel.model == "ollama/llama3:8b"
    assert sel.api_base == "http://ollama.example.internal:11434"
    assert sel.kind == "ollama"


# ---------------------------------------------------------------------------
# EMBEDDINGS: remote endpoint via ollama_endpoints config
# ---------------------------------------------------------------------------

def test_embed_uses_remote_endpoint_when_configured(monkeypatch, tmp_path):
    """_embed_ollama picks the configured remote endpoint URL from OllamaMultiClient.

    Adding a remote endpoint to ollama_endpoints is the supported path for
    embedding via a remote Ollama — no OllamaMultiClient rewrite required.
    """
    from skill_hub.ollama_client import (
        OllamaMultiClient, OllamaClientConfig, OllamaEndpoint, reset_client,
    )

    remote_url = "http://ollama.example.internal:11434"
    remote_ep = OllamaEndpoint(name="remote", url=remote_url, priority=1, enabled=True)
    client = OllamaMultiClient(OllamaClientConfig(endpoints=[remote_ep]))
    reset_client()

    import skill_hub.ollama_client as oc
    monkeypatch.setattr(oc, "_client", client)

    captured: dict = {}

    class _FakeProvider:
        def embed(self, text, *, model, timeout, api_base=None):
            captured["api_base"] = api_base
            captured["model"] = model
            return [0.1, 0.2, 0.3]

    import skill_hub.embeddings as emb
    monkeypatch.setattr(emb, "get_provider", lambda: _FakeProvider())
    monkeypatch.setattr(emb, "_cfg", type("C", (), {"get": lambda self, k: {
        "embed_model": "nomic-embed-text",
        "embedding_backend_priority": ["ollama", "sentence_transformers"],
    }.get(k)})())
    # This test exercises endpoint selection, not the down-gate; force the
    # reachability probe up so a "down" result cached by another test does not
    # short-circuit before the endpoint URL is forwarded.
    import skill_hub.llm.escalation as esc
    monkeypatch.setattr(esc, "ollama_daemon_reachable", lambda *a, **k: True)

    result = emb._embed_ollama("hello world", model="nomic-embed-text")

    assert result == [0.1, 0.2, 0.3]
    assert captured["api_base"] == remote_url, (
        "embed must forward the remote endpoint URL to get_provider().embed()"
    )
    assert captured["model"] == "ollama/nomic-embed-text"
