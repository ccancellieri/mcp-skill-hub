"""#58 (re-scoped): cache-prefix stability — cheap wins.

Two safe improvements to keep injected/reused prompt prefixes cache-stable:

(a) Deterministic ordering — the router's injected ``systemMessage`` emits
    preloaded skill/plugin names in canonical (sorted) order, so the same set
    selected in a different relevance rank produces byte-identical text.
(b) Extended cache TTL — ``_apply_cache_control`` only attaches Anthropic's
    longer ``ttl`` when the operator opts in via ``llm_cache_extended_ttl``;
    otherwise it stays on the default ephemeral window (no API risk).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# (a) deterministic ordering in the injected systemMessage
# ---------------------------------------------------------------------------

def _make_verdict(skills, plugins):
    from skill_hub.router.verdict import Verdict

    # Verdict is a dataclass; construct with the fields the formatter reads.
    return Verdict(
        model="claude-sonnet-4-6",
        confidence=0.9,
        reasoning="r",
        tier_used=1,
        preload_skills=list(skills),
        preload_plugins=list(plugins),
    )


def test_preloaded_skills_emitted_in_sorted_order():
    """The same skill set in different selection order yields identical text."""
    from skill_hub.router.verdict import format_system_message

    a = format_system_message(_make_verdict(["zeta", "alpha", "mu"], []))
    b = format_system_message(_make_verdict(["mu", "zeta", "alpha"], []))
    assert a == b, "injected skill line must be order-independent (sorted)"
    assert "alpha, mu, zeta" in a


def test_preloaded_plugins_emitted_in_sorted_order():
    from skill_hub.router.verdict import format_system_message

    a = format_system_message(_make_verdict([], ["plugin-z", "plugin-a"]))
    b = format_system_message(_make_verdict([], ["plugin-a", "plugin-z"]))
    assert a == b
    assert "plugin-a, plugin-z" in a


# ---------------------------------------------------------------------------
# (b) extended cache TTL is opt-in and gated
# ---------------------------------------------------------------------------

def _provider():
    pytest.importorskip("litellm")
    from skill_hub.llm.litellm_adapter import LitellmProvider

    return LitellmProvider()


def _cache_control_of(messages):
    """Pull the cache_control dict off the marked block, or None."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    return block["cache_control"]
        return None
    return None


def test_extended_ttl_omitted_when_flag_off(tmp_path, monkeypatch):
    """With llm_cache_extended_ttl off, no ttl is attached even if requested."""
    from skill_hub import config as cfg

    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "get", lambda k, default=None: (
        False if k == "llm_cache_extended_ttl" else cfg._DEFAULTS.get(k, default)
    ))

    p = _provider()
    msgs = [{"role": "user", "content": "hello"}]
    out = p._apply_cache_control(msgs, "anthropic/claude-sonnet-4-6", cache_ttl="1h")
    cc = _cache_control_of(out)
    assert cc is not None, "cache_control must still be applied"
    assert cc.get("type") == "ephemeral"
    assert "ttl" not in cc, "ttl must NOT be set when the extended-ttl flag is off"


def test_extended_ttl_applied_when_flag_on(tmp_path, monkeypatch):
    """With the flag on, the requested ttl is attached to the cache breakpoint."""
    from skill_hub import config as cfg

    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "get", lambda k, default=None: (
        True if k == "llm_cache_extended_ttl" else cfg._DEFAULTS.get(k, default)
    ))

    p = _provider()
    msgs = [{"role": "user", "content": "hello"}]
    out = p._apply_cache_control(msgs, "anthropic/claude-sonnet-4-6", cache_ttl="1h")
    cc = _cache_control_of(out)
    assert cc is not None
    assert cc.get("ttl") == "1h"


def test_no_cache_control_for_non_anthropic(tmp_path, monkeypatch):
    """Ollama / non-Anthropic models never get a cache_control marker."""
    from skill_hub import config as cfg

    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "get", lambda k, default=None: (
        True if k == "llm_cache_extended_ttl" else cfg._DEFAULTS.get(k, default)
    ))

    p = _provider()
    msgs = [{"role": "user", "content": "hello"}]
    out = p._apply_cache_control(msgs, "ollama/qwen2.5-coder:3b", cache_ttl="1h")
    assert _cache_control_of(out) is None
