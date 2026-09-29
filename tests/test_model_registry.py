"""Tests for the provider-agnostic model registry.

Covers id normalisation, litellm/static pricing, tier resolution, latest-in-
family detection, and the sync_lineup upgrade path. Config-dependent tests
monkeypatch ``config.get`` / ``config.set`` so nothing touches the real file.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from skill_hub import model_registry as mr  # noqa: E402


@pytest.fixture
def opus_catalogue(monkeypatch):
    """Keep lineup tests independent of LiteLLM's changing model catalogue."""
    rates = {"input_cost_per_token": 0.000005, "output_cost_per_token": 0.000025}
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(model_cost={
        "anthropic/claude-opus-4-6": rates,
        "anthropic/claude-opus-4-8": rates,
    }))
    monkeypatch.setattr(mr, "_STATIC_USD_PER_M", {"claude-opus-4-6": (5.0, 25.0)})


def test_bare_id_strips_prefix_and_suffix():
    assert mr.bare_id("anthropic/claude-opus-4-8@default") == "claude-opus-4-8"
    assert mr.bare_id("vertex_ai/claude-sonnet-4-6@20250929") == "claude-sonnet-4-6"
    assert mr.bare_id("ollama/qwen2.5-coder:3b") == "qwen2.5-coder"
    assert mr.bare_id("claude-haiku-4-5") == "claude-haiku-4-5"


def test_known_models_priced_and_blended():
    # Sonnet's $3/$15 blended (30/70) = 11.4 — stable across static and litellm.
    assert mr.blended_usd_per_m("claude-sonnet-4-6") == 11.4
    for m in ("claude-haiku-4-5", "claude-opus-4-8", "claude-fable-5"):
        assert (mr.blended_usd_per_m(m) or 0) > 0


def test_short_family_aliases_priced():
    for alias in ("haiku", "sonnet", "opus", "fable"):
        assert mr.blended_usd_per_m(alias) is not None


def test_unknown_or_free_model_returns_none():
    assert mr.blended_usd_per_m("ollama/qwen2.5-coder:3b") is None
    assert mr.blended_usd_per_m("totally-made-up-model") is None


def test_resolve_tier(monkeypatch):
    from skill_hub import config

    cfg = {
        "llm_providers": {
            "tier_smart": "anthropic/claude-sonnet-4-6",
            "tier_planner": "anthropic/claude-opus-4-8",
            "tier_cheap": "ollama/qwen2.5-coder:3b",
        },
        "llm_default_tier": "tier_cheap",
    }
    monkeypatch.setattr(config, "get", lambda k: cfg.get(k))
    assert mr.resolve_tier("tier_smart") == "anthropic/claude-sonnet-4-6"
    assert mr.resolve_tier("sonnet") == "anthropic/claude-sonnet-4-6"   # family alias
    assert mr.resolve_tier("opus") == "anthropic/claude-opus-4-8"


def test_resolve_selection_uses_unique_registry_provider_credentials(monkeypatch):
    from skill_hub import config

    values = {
        "llm_provider_registry": [{
            "name": "work-gateway", "kind": "openai_compatible",
            "api_base": "https://gateway.example/v1",
            "api_key": {"source": "inline", "ref": "secret"},
            "models": [{"id": "vendor/model:latest", "tags": ["python"]}],
        }],
        "llm_providers": {"tier_smart": "vendor/model:latest"},
    }
    monkeypatch.setattr(config, "get", lambda key, default=None: values.get(key, default))

    resolved = mr.resolve_model_selection("tier_smart")

    assert resolved.model == "vendor/model:latest"
    assert resolved.requested == "tier_smart"
    assert resolved.provider == "work-gateway"
    assert resolved.kind == "openai_compatible"
    assert resolved.api_base == "https://gateway.example/v1"
    assert resolved.api_key == "secret"


def test_resolve_selection_requires_qualification_for_duplicate_ids(monkeypatch):
    from skill_hub import config

    values = {
        "llm_provider_registry": [
            {"name": name, "kind": "openai_compatible", "enabled": True,
             "api_base": f"https://{name}.example/v1",
             "api_key": {"source": "inline", "ref": f"key-{name}"},
             "models": [{"id": "shared/model"}]}
            for name in ("first", "second")
        ],
        "llm_providers": {},
    }
    monkeypatch.setattr(config, "get", lambda key, default=None: values.get(key, default))

    with pytest.raises(mr.ModelResolutionError, match="ambiguous"):
        mr.resolve_model_selection("shared/model")

    resolved = mr.resolve_model_selection("second::shared/model")
    assert resolved.model == "shared/model"
    assert resolved.provider == "second"
    assert resolved.api_base == "https://second.example/v1"
    assert resolved.api_key == "key-second"


def test_provider_qualification_survives_legacy_tier_lookup(monkeypatch):
    from skill_hub import config

    values = {
        "llm_provider_registry": [
            {"name": name, "kind": "openai_compatible", "enabled": True,
             "api_base": f"https://{name}.example/v1",
             "api_key": {"source": "inline", "ref": f"key-{name}"},
             "models": [{"id": "shared/model"}]}
            for name in ("first", "second")
        ],
        "llm_providers": {"tier_smart": "second::shared/model"},
    }
    monkeypatch.setattr(config, "get", lambda key, default=None: values.get(key, default))

    resolved = mr.resolve_model_selection("tier_smart")

    assert resolved.requested == "tier_smart"
    assert resolved.provider == "second"
    assert resolved.model == "shared/model"


def test_model_options_are_network_free_and_include_legacy_tiers(monkeypatch):
    from skill_hub import config

    values = {
        "llm_provider_registry": [{
            "name": "local", "kind": "ollama", "enabled": True,
            "models": [{"id": "ollama/qwen:7b"}],
        }],
        "llm_providers": {"tier_smart": "custom/frontier-v2"},
    }
    monkeypatch.setattr(config, "get", lambda key, default=None: values.get(key, default))
    monkeypatch.setattr(mr, "_ollama_models", lambda: pytest.fail("model_options must not probe"))

    options = mr.model_options()

    assert {tuple(sorted(row)) for row in options} == {
        tuple(sorted({"id": "ollama/qwen:7b", "provider": "local", "kind": "ollama",
                      "configured": True, "availability": "configured"})),
        tuple(sorted({"id": "custom/frontier-v2", "provider": "legacy:tier_smart",
                      "kind": "unknown", "configured": True,
                      "availability": "configured"})),
    }


def test_model_options_deduplicates_provider_qualified_saved_models(monkeypatch):
    from skill_hub import config

    values = {
        "llm_provider_registry": [
            {"name": "local", "kind": "ollama", "enabled": True,
             "models": [{"id": "qwen-local"}]},
            {"name": "work", "kind": "openai_compatible", "enabled": True,
             "models": [{"id": "grok"}]},
        ],
        "llm_providers": {
            "tier_cheap": "local::qwen-local",
            "tier_smart": "work::grok",
            "tier_planner": "missing::frontier",
        },
    }
    monkeypatch.setattr(config, "get", lambda key, default=None: values.get(key, default))

    options = mr.model_options()

    assert [(row["provider"], row["id"]) for row in options].count(("local", "qwen-local")) == 1
    assert [(row["provider"], row["id"]) for row in options].count(("work", "grok")) == 1
    assert not any(row["provider"].startswith("legacy:") and row["id"] in {
        "local::qwen-local", "work::grok",
    } for row in options)
    assert any(row["provider"] == "legacy:tier_planner" and
               row["id"] == "missing::frontier" for row in options)


def test_latest_in_family(opus_catalogue):
    latest = mr.latest_in_family("opus")
    assert latest == "claude-opus-4-8"


def test_latest_in_family_orders_numeric_versions_in_mixed_catalogue(monkeypatch):
    rates = {"input_cost_per_token": 0.000005, "output_cost_per_token": 0.000025}
    catalogue = {
        "anthropic/claude-opus-4-9": rates,
        "anthropic/claude-opus-4-10": rates,
        "anthropic/claude-sonnet-9-9": rates,
    }
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(model_cost=catalogue))
    monkeypatch.setattr(mr, "_STATIC_USD_PER_M", {"claude-opus-4-8": (5.0, 25.0)})

    assert mr.latest_in_family("opus") == "claude-opus-4-10"
    catalogue["anthropic/claude-opus-5-1"] = rates
    assert mr.latest_in_family("opus") == "claude-opus-5-1"


def test_sync_lineup_dry_run_detects_stale_and_persists_nothing(monkeypatch, opus_catalogue):
    from skill_hub import config

    cfg = {
        "llm_providers": {
            "tier_planner": "anthropic/claude-opus-4-6",   # stale
            "tier_cheap": "ollama/qwen2.5-coder:3b",        # non-Claude, untouched
        },
        "llm_default_tier": "tier_cheap",
    }
    saved: dict = {}
    monkeypatch.setattr(config, "get", lambda k: cfg.get(k))
    monkeypatch.setattr(config, "set", lambda k, v: saved.update({k: v}))

    res = mr.sync_lineup(dry_run=True)
    planner_changes = [c for c in res["changes"] if c["tier"] == "tier_planner"]
    assert planner_changes and planner_changes[0]["from"] == "anthropic/claude-opus-4-6"
    assert planner_changes[0]["to"] == "anthropic/claude-opus-4-8"
    assert res["applied"] is False
    assert not saved, "dry-run must not persist"
    # Non-Claude tier never proposed for change.
    assert all(c["tier"] != "tier_cheap" for c in res["changes"])


def test_sync_lineup_applies_and_persists(monkeypatch, opus_catalogue):
    from skill_hub import config

    cfg = {
        "llm_providers": {"tier_planner": "anthropic/claude-opus-4-6"},
        "llm_default_tier": "tier_cheap",
    }
    saved: dict = {}
    monkeypatch.setattr(config, "get", lambda k: cfg.get(k))
    monkeypatch.setattr(config, "set", lambda k, v: saved.update({k: v}))

    res = mr.sync_lineup(dry_run=False)
    assert res["applied"] is True
    assert "llm_providers" in saved
    assert saved["llm_providers"]["tier_planner"] == "anthropic/claude-opus-4-8"
