"""Tests for the router's compress stage (WS-A) — ``router/route._compress_stage``.

The stage estimates tokens as ``len(text)//4`` against
``router_compress_budget_tokens`` and, when over budget, delegates to
``compression.maybe_compress()`` — deterministic-only (JSON minify /
duplicate-line collapse), never the retired lossy ML path.
"""
from __future__ import annotations

from skill_hub.router.route import _compress_stage


def _cfg(**overrides) -> dict:
    base = {"router_compress_context_enabled": True, "router_compress_budget_tokens": 1500}
    base.update(overrides)
    return base


def _isolate_config(monkeypatch, tmp_path):
    """Point config.CONFIG_PATH at an empty file so maybe_compress()'s own
    compression_enabled/compression_min_tokens gates read pure defaults,
    independent of any config the rest of the suite persisted."""
    from skill_hub import config

    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "cfg.json")


def test_compress_stage_over_budget_compresses_and_shortens(monkeypatch, tmp_path):
    _isolate_config(monkeypatch, tmp_path)

    marker = "<<ccr:deadbeef>>"
    # Duplicate-line block the deterministic DEDUP strategy shrinks, well over
    # both the router budget and compression's own min-token gate.
    big = "\n".join(["skill hint: foo bar baz"] * 200) + f"\n{marker}\n"
    output = {"systemMessage": big}

    result = _compress_stage(output, _cfg(router_compress_budget_tokens=50))

    assert len(result["systemMessage"]) < len(big)
    assert "(x200)" in result["systemMessage"]  # deterministic DEDUP, not lossy ML
    assert marker in result["systemMessage"]     # reversible marker intact


def test_compress_stage_under_budget_unchanged(monkeypatch, tmp_path):
    _isolate_config(monkeypatch, tmp_path)

    small_system = "short systemMessage, well under budget"
    small_user = "also short"
    output = {"systemMessage": small_system, "userMessage": small_user}

    result = _compress_stage(output, _cfg(router_compress_budget_tokens=1500))

    assert result["systemMessage"] == small_system
    assert result["userMessage"] == small_user


def test_compress_stage_disabled_is_noop(monkeypatch, tmp_path):
    _isolate_config(monkeypatch, tmp_path)

    big = "\n".join(["skill hint: foo bar baz"] * 200)
    output = {"systemMessage": big}

    result = _compress_stage(
        output, _cfg(router_compress_context_enabled=False, router_compress_budget_tokens=10)
    )

    assert result["systemMessage"] == big


def test_compress_stage_respects_global_compression_enabled_gate(monkeypatch, tmp_path):
    """Even when the stage itself is on and over budget, the existing
    compression_enabled master switch must still be honored (delegated to
    maybe_compress, not duplicated by the stage)."""
    from skill_hub import config

    _isolate_config(monkeypatch, tmp_path)
    monkeypatch.setattr(
        config, "get",
        lambda k: False if k == "compression_enabled" else config._DEFAULTS.get(k),
    )

    big = "\n".join(["skill hint: foo bar baz"] * 200)
    output = {"systemMessage": big}

    result = _compress_stage(output, _cfg(router_compress_budget_tokens=10))

    assert result["systemMessage"] == big


def test_compress_stage_missing_fields_are_noop(monkeypatch, tmp_path):
    _isolate_config(monkeypatch, tmp_path)

    output: dict = {}
    result = _compress_stage(output, _cfg(router_compress_budget_tokens=1))

    assert result == {}
