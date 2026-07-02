"""Tests for the Ollama down-gate: skip calls when daemon is known-down.

Covers:
(a) Chat ladder: when ollama_daemon_reachable() is False, ladder skips ollama
    models without issuing any HTTP call to litellm.
(b) Embed path: when ollama_daemon_reachable() is False, embed() does NOT
    attempt _embed_ollama's HTTP call and falls back to sentence_transformers.
(c) A gated skip (daemon down) writes no llm_call error event to the store.
(d) The "down" probe is cached for ollama_down_probe_ttl_seconds, not the
    shorter up-TTL, so bursts do not re-probe.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Shared registry for chat-path tests (no ollama in registry → gateway only)
# ---------------------------------------------------------------------------

_REG_GATEWAY_ONLY = {
    "llm_metering_enabled": False,
    "llm_provider_registry": [
        {
            "name": "gw",
            "kind": "openai_compatible",
            "api_base": "https://gw/v1",
            "api_key": {"source": "inline", "ref": "sk"},
            "enabled": True,
            "order": 30,
            "models": [{"id": "gw-model", "complexity": "light", "tags": ["fast", "digest"]}],
        },
    ],
}

_REG_LOCAL_PLUS_GW = {
    "llm_metering_enabled": False,
    "llm_provider_registry": [
        {
            "name": "local",
            "kind": "ollama",
            "api_base": "",
            "api_key": {},
            "enabled": True,
            "order": 10,
            "models": [{"id": "ollama/local-model", "complexity": "light", "tags": ["fast", "digest"]}],
        },
        {
            "name": "gw",
            "kind": "openai_compatible",
            "api_base": "https://gw/v1",
            "api_key": {"source": "inline", "ref": "sk"},
            "enabled": True,
            "order": 30,
            "models": [{"id": "gw-model", "complexity": "light", "tags": ["fast", "digest"]}],
        },
    ],
}


def _write_cfg(monkeypatch, tmp_path, data: dict):
    import skill_hub.config as cfg
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data))
    monkeypatch.setattr(cfg, "CONFIG_PATH", p)


def _fake_litellm_ok(calls: list[str]):
    class FakeLitellm:
        suppress_debug_info = True
        drop_params = True

        def completion(self, **kwargs):
            calls.append(kwargs["model"])
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    return FakeLitellm()


# ---------------------------------------------------------------------------
# (a) Chat ladder — daemon down → no ollama HTTP attempt, routes to gateway
# ---------------------------------------------------------------------------

def test_chat_ladder_skips_ollama_when_daemon_down(monkeypatch, tmp_path):
    """When daemon is down, the chat ladder must not issue any ollama call."""
    import importlib
    _write_cfg(monkeypatch, tmp_path, _REG_LOCAL_PLUS_GW)

    import skill_hub.llm.escalation as escalation
    import skill_hub.llm.litellm_adapter as litellm_adapter
    import skill_hub.llm.provider as provider
    for m in (escalation, litellm_adapter, provider):
        importlib.reload(m)
    escalation.reset_cooldowns()
    escalation.reset_reachability()

    # Force daemon-down without touching the network.
    monkeypatch.setattr(escalation, "ollama_daemon_reachable", lambda **k: False)

    calls: list[str] = []
    p = litellm_adapter.LitellmProvider()
    p._litellm = _fake_litellm_ok(calls)

    out = p.complete("hello", op="compact")   # "compact" is in _OP_ROUTING → ladder
    assert out == "ok"
    # The local ollama model must never appear in calls.
    assert all("ollama" not in c for c in calls), f"unexpected ollama call: {calls}"
    # Gateway was reached.
    assert any("gw-model" in c for c in calls), f"gateway not called: {calls}"


def test_chat_pinned_local_skips_http_when_daemon_down(monkeypatch, tmp_path):
    """A pinned local model must skip the HTTP call and route to ladder when down."""
    import importlib
    _write_cfg(monkeypatch, tmp_path, _REG_LOCAL_PLUS_GW)

    import skill_hub.llm.escalation as escalation
    import skill_hub.llm.litellm_adapter as litellm_adapter
    import skill_hub.llm.provider as provider
    for m in (escalation, litellm_adapter, provider):
        importlib.reload(m)
    escalation.reset_cooldowns()
    escalation.reset_reachability()

    monkeypatch.setattr(escalation, "ollama_daemon_reachable", lambda **k: False)

    calls: list[str] = []
    p = litellm_adapter.LitellmProvider()
    p._litellm = _fake_litellm_ok(calls)

    out = p.complete("hello", model="ollama/local-model", op="compact")
    assert out == "ok"
    # Local model must not be called.
    assert "ollama/local-model" not in calls, f"doomed ollama call issued: {calls}"
    assert any("gw-model" in c for c in calls)


# ---------------------------------------------------------------------------
# (b) Embed path — daemon down → _embed_ollama is skipped, ST used instead
# ---------------------------------------------------------------------------

def test_embed_skips_ollama_http_when_daemon_down(monkeypatch):
    """embed() must not call get_provider().embed() (the HTTP call) when the
    daemon is down: the gate in _embed_ollama must raise before reaching it,
    and the cascade must fall through to sentence_transformers.
    """
    import skill_hub.embeddings as emb
    import skill_hub.llm.escalation as escalation

    escalation.reset_reachability()

    _FAKE_ST_VEC = [0.5, 0.6, 0.7]
    http_attempted = {"v": False}

    # Patch the provider so we can detect if the HTTP embed call was attempted.
    provider_mock = MagicMock()
    def _provider_embed(*a, **k):
        http_attempted["v"] = True
        raise Exception("should not be called")
    provider_mock.embed.side_effect = _provider_embed

    original_st = emb._embed_sentence_transformers
    emb._embed_sentence_transformers = lambda text: _FAKE_ST_VEC

    try:
        # _embed_ollama imports ollama_daemon_reachable directly from the module;
        # patch it at its definition site so the import inside the function sees False.
        monkeypatch.setattr(escalation, "ollama_daemon_reachable", lambda **k: False)
        with patch("skill_hub.embeddings.get_provider", return_value=provider_mock), \
             patch("skill_hub.embeddings._cfg") as cfg_mock:
            cfg_mock.get.side_effect = lambda k: {
                "embedding_backend_priority": ["ollama", "sentence_transformers"],
                "embed_model": "nomic-embed-text",
            }.get(k)
            result = emb.embed("test text")
    finally:
        emb._embed_sentence_transformers = original_st
        escalation.reset_reachability()

    # The HTTP embed call was never reached.
    assert http_attempted["v"] is False, "HTTP attempt was made despite daemon being down"
    # Fell back to sentence_transformers.
    assert result == _FAKE_ST_VEC


def test_embed_uses_ollama_when_daemon_up(monkeypatch):
    """embed() must use ollama when daemon is reachable — no regression."""
    import skill_hub.embeddings as emb
    import skill_hub.llm.escalation as escalation

    escalation.reset_reachability()

    _FAKE_OLLAMA_VEC = [0.1, 0.2, 0.3]
    original_embed_ollama = emb._embed_ollama

    def patched_embed_ollama(text, *, model, timeout=15.0):
        return _FAKE_OLLAMA_VEC

    emb._embed_ollama = patched_embed_ollama

    try:
        monkeypatch.setattr(escalation, "ollama_daemon_reachable", lambda **k: True)
        with patch("skill_hub.embeddings._cfg") as cfg_mock:
            cfg_mock.get.side_effect = lambda k: {
                "embedding_backend_priority": ["ollama", "sentence_transformers"],
                "embed_model": "nomic-embed-text",
            }.get(k)
            result = emb.embed("test text")
    finally:
        emb._embed_ollama = original_embed_ollama
        escalation.reset_reachability()

    assert result == _FAKE_OLLAMA_VEC


# ---------------------------------------------------------------------------
# (c) Gated skip writes no llm_call error event
# ---------------------------------------------------------------------------

def test_gated_skip_writes_no_error_event(monkeypatch, tmp_path):
    """When the daemon-down gate fires, no llm_call error event must be written."""
    import importlib
    _write_cfg(monkeypatch, tmp_path, {**_REG_LOCAL_PLUS_GW, "llm_metering_enabled": True})

    import skill_hub.llm.escalation as escalation
    import skill_hub.llm.litellm_adapter as litellm_adapter
    import skill_hub.llm.provider as provider
    for m in (escalation, litellm_adapter, provider):
        importlib.reload(m)
    escalation.reset_cooldowns()
    escalation.reset_reachability()

    monkeypatch.setattr(escalation, "ollama_daemon_reachable", lambda **k: False)

    emitted: list[dict] = []

    class FakeStore:
        def append_event(self, *, session_id="", kind="", payload=None, tool_name=None):
            emitted.append({"kind": kind, "payload": payload})

    import skill_hub.store as store_mod
    monkeypatch.setattr(store_mod, "get_store", lambda: FakeStore())

    calls: list[str] = []
    p = litellm_adapter.LitellmProvider()
    p._litellm = _fake_litellm_ok(calls)

    out = p.complete("hello", model="ollama/local-model", op="compact")
    assert out == "ok"

    # No error events must appear for ollama.
    ollama_errors = [
        e for e in emitted
        if e["kind"] == "llm_call"
        and (e.get("payload") or {}).get("status") == "error"
        and "ollama" in str((e.get("payload") or {}).get("model", ""))
    ]
    assert not ollama_errors, f"unexpected ollama error events: {ollama_errors}"


# ---------------------------------------------------------------------------
# (d) Down probe is cached for ollama_down_probe_ttl_seconds
# ---------------------------------------------------------------------------

def test_down_probe_cached_for_configured_ttl(monkeypatch, tmp_path):
    """A False probe result must be cached for ollama_down_probe_ttl_seconds,
    so a second call within that window does not re-probe."""
    import skill_hub.config as cfg
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"ollama_down_probe_ttl_seconds": 300}))
    monkeypatch.setattr(cfg, "CONFIG_PATH", p)

    from skill_hub.llm import escalation
    escalation.reset_reachability()

    # get_ollama_client is imported inside ollama_daemon_reachable with a local
    # ``from ..ollama_client import get_ollama_client`` — patch it at the source module.
    with patch("skill_hub.ollama_client.get_ollama_client") as mock_client:
        mock_client.return_value.get_api_base.return_value = None  # no endpoint → ok=False
        # First call — cache is empty, so the probe runs.
        r1 = escalation.ollama_daemon_reachable()
        assert r1 is False

        # Confirm the cache entry was written with the long TTL from config.
        cached = escalation._REACH_CACHE.get("ollama")
        assert cached is not None
        expiry, val = cached
        assert val is False
        # TTL from config is 300s; cache expiry must be at least 290s in the future.
        assert expiry - time.time() >= 290, f"expected ~300s cache TTL, got {expiry - time.time():.1f}s"

        # Second call within the cache window — must NOT re-probe.
        r2 = escalation.ollama_daemon_reachable()
        assert r2 is False
        # mock_client was called exactly once (the second call was a cache hit).
        assert mock_client.call_count == 1, (
            f"probe was re-issued ({mock_client.call_count} calls) despite valid cache"
        )

    escalation.reset_reachability()
