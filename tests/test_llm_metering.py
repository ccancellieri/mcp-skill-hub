"""Tests for per-call local-LLM metering (issue #59).

Covers:
(a) get_llm_stats() returns all documented keys on an empty store.
(b) After two append_event calls (one ok, one error), get_llm_stats() aggregates
    correctly: calls==2, errors==1, token totals, by_op/by_model populated, avg_latency_ms.
(c) _emit_llm_event respects llm_metering_enabled=False — no event appended.

No ML model is loaded. No network calls are made.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_store(tmp_path, monkeypatch):
    """A fresh SkillStore backed by a temp DB that does not touch the live DB."""
    from skill_hub.store import SkillStore

    db_path = tmp_path / "test_llm_metering.db"
    monkeypatch.setattr("skill_hub.store.DB_PATH", db_path)
    store = SkillStore(db_path=db_path)
    yield store
    store.close()


@pytest.fixture()
def patched_store(isolated_store, monkeypatch):
    """Redirect get_store() to the isolated fixture store."""
    import skill_hub.store as _store_mod

    monkeypatch.setattr(_store_mod, "_default_store", isolated_store)
    monkeypatch.setattr(_store_mod, "get_store", lambda: isolated_store)
    return isolated_store


# ---------------------------------------------------------------------------
# (a) get_llm_stats() returns all documented keys on an empty store
# ---------------------------------------------------------------------------

_EXPECTED_STATS_KEYS = {
    "calls", "errors",
    "total_duration_ms", "avg_latency_ms",
    "prompt_tokens", "completion_tokens", "total_tokens",
    "tokens_per_sec",
    "by_op", "by_model",
}


def test_get_llm_stats_returns_documented_keys(isolated_store):
    """get_llm_stats() must always return a dict with every documented key,
    even when there are no events."""
    stats = isolated_store.get_llm_stats()
    assert isinstance(stats, dict)
    missing = _EXPECTED_STATS_KEYS - stats.keys()
    assert not missing, f"get_llm_stats() is missing keys: {missing}"


def test_get_llm_stats_empty_store_zeroes(isolated_store):
    """On an empty store every numeric field is 0 / 0.0 and dicts are empty."""
    s = isolated_store.get_llm_stats()
    assert s["calls"] == 0
    assert s["errors"] == 0
    assert s["total_duration_ms"] == 0
    assert s["avg_latency_ms"] == 0.0
    assert s["prompt_tokens"] == 0
    assert s["completion_tokens"] == 0
    assert s["total_tokens"] == 0
    assert s["tokens_per_sec"] == 0.0
    assert s["by_op"] == {}
    assert s["by_model"] == {}


# ---------------------------------------------------------------------------
# (b) Aggregation after two events (one ok, one error)
# ---------------------------------------------------------------------------

def test_get_llm_stats_aggregates_correctly(isolated_store):
    """After one ok and one error event, aggregation must be correct."""
    isolated_store.append_event(
        session_id="",
        kind="llm_call",
        payload={
            "op": "compact",
            "model": "ollama/deepseek-r1:1.5b",
            "tier": "tier_cheap",
            "duration_ms": 400,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "status": "ok",
        },
    )
    isolated_store.append_event(
        session_id="",
        kind="llm_call",
        payload={
            "op": "rerank",
            "model": "ollama/deepseek-r1:1.5b",
            "tier": "tier_cheap",
            "duration_ms": 200,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "status": "error",
        },
    )

    s = isolated_store.get_llm_stats()

    # Top-level aggregates
    assert s["calls"] == 2
    assert s["errors"] == 1
    assert s["total_duration_ms"] == 600
    # avg latency: 600 / 2 = 300
    assert s["avg_latency_ms"] == pytest.approx(300.0, rel=0.01)
    # token totals: only the ok event contributed real tokens
    assert s["prompt_tokens"] == 100
    assert s["completion_tokens"] == 50
    assert s["total_tokens"] == 150

    # by_op must have both ops
    assert "compact" in s["by_op"]
    assert "rerank" in s["by_op"]
    assert s["by_op"]["compact"]["count"] == 1
    assert s["by_op"]["compact"]["total_tokens"] == 150
    assert s["by_op"]["rerank"]["count"] == 1
    assert s["by_op"]["rerank"]["total_tokens"] == 0

    # by_model must have the model
    model_key = "ollama/deepseek-r1:1.5b"
    assert model_key in s["by_model"]
    assert s["by_model"][model_key]["count"] == 2
    assert s["by_model"][model_key]["total_tokens"] == 150


# ---------------------------------------------------------------------------
# (c) _emit_llm_event respects llm_metering_enabled=False
# ---------------------------------------------------------------------------

def test_emit_llm_event_respects_metering_disabled_flag(patched_store, tmp_path, monkeypatch):
    """When llm_metering_enabled=False, _emit_llm_event must not append any event."""
    from skill_hub import config as cfg

    # Isolate config writes to a temp file
    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "config.json")

    # Patch cfg_get inside litellm_adapter to return False for llm_metering_enabled
    import skill_hub.llm.litellm_adapter as adapter_mod

    def _cfg_get_disabled(key, default=None):
        if key == "llm_metering_enabled":
            return False
        return cfg._DEFAULTS.get(key, default)

    # Monkeypatch the config.get used inside _emit_llm_event by patching
    # the config module's get function as seen from the adapter module.
    monkeypatch.setattr(cfg, "get", _cfg_get_disabled)

    before = len(patched_store.get_events(kind="llm_call"))

    adapter_mod._emit_llm_event(
        op="test_op",
        model="ollama/test-model",
        tier="tier_cheap",
        duration_ms=100,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        status="ok",
    )

    after = len(patched_store.get_events(kind="llm_call"))
    assert after == before, (
        f"Expected no new llm_call event when metering is disabled; "
        f"count before={before}, after={after}"
    )
