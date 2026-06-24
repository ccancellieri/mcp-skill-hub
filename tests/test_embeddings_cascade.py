"""Tests for embed() cascade: Ollama → SentenceTransformers."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_VEC = [0.1, 0.2, 0.3]


def _make_provider_mock(vec=_FAKE_VEC):
    """Return a mock LLM provider whose .embed() returns *vec*."""
    mock = MagicMock()
    mock.embed.return_value = vec
    return mock


# ---------------------------------------------------------------------------
# embed() — ollama succeeds (first in priority)
# ---------------------------------------------------------------------------


def test_embed_ollama_first_success(monkeypatch):
    """embed() returns result from ollama when an endpoint is healthy."""
    provider_mock = _make_provider_mock()

    with patch("skill_hub.embeddings.get_provider", return_value=provider_mock), \
         patch("skill_hub.embeddings._cfg") as cfg_mock:

        cfg_mock.get.side_effect = lambda k: {
            "embedding_backend_priority": ["ollama", "sentence_transformers"],
            "embed_model": "nomic-embed-text",
            "sentence_transformers_model": "all-MiniLM-L6-v2",
        }.get(k)

        import skill_hub.embeddings as emb

        original_ollama = emb._embed_ollama

        def fake_ollama(text, *, model, timeout=15.0):
            return _FAKE_VEC

        emb._embed_ollama = fake_ollama
        try:
            result = emb.embed("hello world")
        finally:
            emb._embed_ollama = original_ollama

    assert result == _FAKE_VEC


# ---------------------------------------------------------------------------
# embed() — ollama unavailable → falls back to sentence_transformers
# ---------------------------------------------------------------------------


def test_embed_falls_back_to_sentence_transformers(monkeypatch):
    """embed() uses sentence_transformers when ollama fails."""
    import skill_hub.embeddings as emb

    original_ollama = emb._embed_ollama
    original_st = emb._embed_sentence_transformers

    def fake_ollama(text, *, model, timeout=15.0):
        raise RuntimeError("no healthy Ollama endpoint available")

    def fake_st(text):
        return [0.9, 0.8, 0.7]

    emb._embed_ollama = fake_ollama
    emb._embed_sentence_transformers = fake_st

    try:
        with patch("skill_hub.embeddings._cfg") as cfg_mock:
            cfg_mock.get.side_effect = lambda k: {
                "embedding_backend_priority": ["ollama", "sentence_transformers"],
            }.get(k)

            result = emb.embed("hello")
    finally:
        emb._embed_ollama = original_ollama
        emb._embed_sentence_transformers = original_st

    assert result == [0.9, 0.8, 0.7]


# ---------------------------------------------------------------------------
# embed() — all backends fail → RuntimeError
# ---------------------------------------------------------------------------


def test_embed_raises_when_all_backends_fail(monkeypatch):
    """embed() raises RuntimeError listing all backend errors when all fail."""
    import skill_hub.embeddings as emb

    original_ollama = emb._embed_ollama
    original_st = emb._embed_sentence_transformers

    def fake_ollama(text, *, model, timeout=15.0):
        raise RuntimeError("no healthy Ollama endpoint available")

    def fake_st(text):
        raise RuntimeError("sentence_transformers not installed")

    emb._embed_ollama = fake_ollama
    emb._embed_sentence_transformers = fake_st

    try:
        with patch("skill_hub.embeddings._cfg") as cfg_mock:
            cfg_mock.get.side_effect = lambda k: {
                "embedding_backend_priority": ["ollama", "sentence_transformers"],
            }.get(k)

            with pytest.raises(RuntimeError, match="all embedding backends failed"):
                emb.embed("hello")
    finally:
        emb._embed_ollama = original_ollama
        emb._embed_sentence_transformers = original_st


# ---------------------------------------------------------------------------
# ollama_available() — returns False when no healthy endpoints
# ---------------------------------------------------------------------------


def test_ollama_available_returns_false_when_no_endpoints():
    """ollama_available() returns False when OllamaMultiClient has no healthy endpoint."""
    ollama_client_mock = MagicMock()
    ollama_client_mock.get_api_base.return_value = None  # no healthy endpoint

    with patch("skill_hub.embeddings.get_ollama_client", return_value=ollama_client_mock, create=True):
        # Need to inject get_ollama_client into embeddings namespace
        import skill_hub.embeddings as emb
        original = getattr(emb, "get_ollama_client", None)
        emb.get_ollama_client = lambda: ollama_client_mock  # type: ignore[attr-defined]
        try:
            result = emb.ollama_available("nomic-embed-text")
        finally:
            if original is None:
                try:
                    delattr(emb, "get_ollama_client")
                except AttributeError:
                    pass
            else:
                emb.get_ollama_client = original  # type: ignore[attr-defined]

    assert result is False


def test_ollama_available_returns_false_when_no_endpoints_via_import():
    """ollama_available() returns False via normal import path when no endpoints."""
    from skill_hub.ollama_client import OllamaMultiClient, OllamaClientConfig

    # Build a client with no endpoints
    empty_config = OllamaClientConfig(endpoints=[])
    empty_client = OllamaMultiClient(empty_config)

    with patch("skill_hub.ollama_client.get_ollama_client", return_value=empty_client):
        import skill_hub.embeddings as emb
        # Temporarily reset singleton so our mock is used
        import skill_hub.ollama_client as oc
        original_client = oc._client
        oc._client = empty_client
        try:
            result = emb.ollama_available("nomic-embed-text")
        finally:
            oc._client = original_client

    assert result is False
