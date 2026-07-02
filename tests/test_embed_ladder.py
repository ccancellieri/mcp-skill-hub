"""Embed lane over the provider ladder (#134).

Covers: registry parsing of the ``embed`` flag, chat/embed lane separation,
``select_embed`` walking, the ``_embed_ladder`` HTTP backend with dim guard
and cooldown rotation, and ``embed_available`` recognising a remote embed
provider.
"""
from __future__ import annotations

import json

import pytest

_REG_EMBED = {
    "llm_metering_enabled": False,
    "embed_model": "nomic-embed-text",
    "llm_provider_registry": [
        {
            "name": "local-models",
            "kind": "ollama",
            "enabled": True,
            "order": 10,
            "models": [
                {"id": "ollama/qwen", "complexity": "light"},
                {"id": "nomic-embed-text", "embed": True},
            ],
        },
        {
            "name": "remote-ollama",
            "kind": "ollama",
            "api_base": "http://remote-host:11434",
            "enabled": True,
            "order": 20,
            "models": [{"id": "nomic-embed-text", "embed": True}],
        },
        {
            "name": "remote-models",
            "kind": "openai_compatible",
            "api_base": "https://gw/v1",
            "api_key": {"source": "inline", "ref": "sk"},
            "enabled": True,
            "order": 30,
            "models": [
                {"id": "gw-chat", "complexity": "light"},
                {"id": "gw-embed", "embed": True},
            ],
        },
        {
            "name": "main-models",
            "kind": "anthropic",
            "personal": True,
            "api_key": {"source": "inline", "ref": "sk2"},
            "enabled": True,
            "order": 90,
            "models": [{"id": "anthropic/claude-haiku-4-5", "embed": True}],
        },
    ],
}


@pytest.fixture()
def escalation(monkeypatch, tmp_path):
    import skill_hub.config as cfg
    p = tmp_path / "config.json"
    p.write_text(json.dumps(_REG_EMBED))
    monkeypatch.setattr(cfg, "CONFIG_PATH", p)
    import skill_hub.llm.escalation as esc
    esc.reset_cooldowns()
    yield esc
    esc.reset_cooldowns()


def test_registry_parses_embed_flag(escalation):
    from skill_hub.llm.registry import load_registry
    provs = {p.name: p for p in load_registry()}
    assert any(m.embed for m in provs["remote-models"].models)
    assert not next(m for m in provs["remote-models"].models if m.id == "gw-chat").embed


def test_chat_selection_never_picks_embed_models(escalation):
    # Chat walk over local-models must return the chat model, not the embed one.
    sel = escalation.select(0.2)
    assert sel is not None
    assert sel.model == "ollama/qwen"


def test_select_embed_prefers_first_remote_capable(escalation):
    # Local ollama (no api_base) is skipped — remote ollama host wins by order.
    sel = escalation.select_embed()
    assert sel is not None
    assert sel.provider == "remote-ollama"
    assert sel.model == "nomic-embed-text"


def test_select_embed_rotates_on_exclude_and_skips_anthropic(escalation):
    sel = escalation.select_embed(exclude={"nomic-embed-text"})
    assert sel is not None and sel.provider == "remote-models"
    # Excluding everything reachable → None (anthropic never selected).
    sel2 = escalation.select_embed(exclude={"nomic-embed-text", "gw-embed"})
    assert sel2 is None


def test_select_embed_respects_cooldown(escalation):
    escalation.mark_cooldown("nomic-embed-text", seconds=60)
    sel = escalation.select_embed()
    assert sel is not None and sel.provider == "remote-models"


def test_has_ladder_embed_provider(escalation):
    assert escalation.has_ladder_embed_provider() is True


def test_embed_ladder_openai_compatible_and_dim_guard(escalation, monkeypatch):
    import skill_hub.embeddings as emb

    calls: list[str] = []

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(url)
        if "remote-host" in url:
            # Remote ollama returns a wrong-dimension vector → rejected.
            return FakeResp({"embeddings": [[0.1] * 384]})
        return FakeResp({"data": [{"embedding": [0.5] * 768}]})

    monkeypatch.setattr(emb.httpx, "post", fake_post)
    vec = emb._embed_ladder("hello")
    assert len(vec) == 768
    assert any("remote-host" in c for c in calls)      # tried, rejected on dim
    assert any("gw/v1/embeddings" in c for c in calls)  # rotated to gateway


def test_embed_ladder_cools_failed_model_and_raises_when_exhausted(
        escalation, monkeypatch):
    import skill_hub.embeddings as emb

    def fake_post(url, json=None, headers=None, timeout=None):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(emb.httpx, "post", fake_post)
    with pytest.raises(RuntimeError, match="no ladder embed provider usable"):
        emb._embed_ladder("hello")
    assert escalation.is_cooled("gw-embed")


def test_embed_cascade_falls_through_to_ladder(escalation, monkeypatch):
    import skill_hub.embeddings as emb

    # Default priority is ["ollama", "ladder", "sentence_transformers"].
    monkeypatch.setattr(emb, "_embed_ollama", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("daemon down")))
    monkeypatch.setattr(emb, "_embed_ladder", lambda text, timeout=15.0: [0.5] * 768)
    monkeypatch.setattr(emb, "_hot_path", lambda: False)

    vec = emb.embed("hello")
    assert len(vec) == 768


def test_embed_available_true_with_only_remote_embed_provider(
        escalation, monkeypatch):
    import skill_hub.embeddings as emb
    from skill_hub import ollama_client

    class NoOllama:
        def get_api_base(self, model):
            return None

    monkeypatch.setattr(ollama_client, "get_ollama_client", lambda: NoOllama())
    monkeypatch.setattr(emb, "_hot_path", lambda: False)
    assert emb.embed_available() is True


def test_hot_path_strips_ladder_backend(escalation, monkeypatch):
    """Per-prompt hooks must never pay a remote embed round-trip."""
    import skill_hub.embeddings as emb

    monkeypatch.setattr(emb, "_hot_path", lambda: True)
    called: list[str] = []
    monkeypatch.setattr(emb, "_embed_ollama", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("daemon down")))
    monkeypatch.setattr(
        emb, "_embed_ladder",
        lambda *a, **k: called.append("ladder") or [0.5] * 768)

    with pytest.raises(RuntimeError, match="all embedding backends failed"):
        emb.embed("hello")
    assert not called
