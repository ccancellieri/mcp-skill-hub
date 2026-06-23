"""Tests for the provider-agnostic model registry.

Covers id normalisation, litellm/static pricing, tier resolution, latest-in-
family detection, and the sync_lineup upgrade path. Config-dependent tests
monkeypatch ``config.get`` / ``config.set`` so nothing touches the real file.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from skill_hub import model_registry as mr  # noqa: E402


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


def test_latest_in_family():
    latest = mr.latest_in_family("opus")
    assert latest is not None and latest.startswith("claude-opus-4")
    # 4-8 must beat 4-6 in version ordering.
    assert latest >= "claude-opus-4-8"


def test_sync_lineup_dry_run_detects_stale_and_persists_nothing(monkeypatch):
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
    assert planner_changes[0]["to"].startswith("anthropic/claude-opus-4")
    assert res["applied"] is False
    assert not saved, "dry-run must not persist"
    # Non-Claude tier never proposed for change.
    assert all(c["tier"] != "tier_cheap" for c in res["changes"])


def test_sync_lineup_applies_and_persists(monkeypatch):
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
    assert saved["llm_providers"]["tier_planner"].startswith("anthropic/claude-opus-4")
