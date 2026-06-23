"""Tests for the per-session routing-verdict cache (issue #88).

Convention: isolate CONFIG_PATH to a tmp path BEFORE importing skill_hub so
tests never touch ~/.claude/mcp-skill-hub/config.json.  The cache dir is
redirected via monkeypatching verdict_cache.cache_dir / verdict_cache.cache_path.

Does NOT import skill_hub.server (avoids the live-DB module-level side-effect).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_verdict(skills=None, plugins=None):
    from skill_hub.router.verdict import Verdict

    return Verdict(
        model="sonnet",
        confidence=0.9,
        reasoning="test",
        tier_used=1,
        preload_skills=list(skills or []),
        preload_plugins=list(plugins or []),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    """Point CONFIG_PATH at a tmp file so no real config is read or written."""
    from skill_hub import config as cfg

    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "config.json")
    cfg.reset_vault_migration_flag()
    yield


@pytest.fixture()
def tmp_cache_dir(tmp_path, monkeypatch):
    """Redirect cache_dir() and cache_path() to a temp directory."""
    import skill_hub.router.verdict_cache as vc

    cache_root = tmp_path / "verdict-cache"

    monkeypatch.setattr(vc, "cache_dir", lambda: cache_root)
    monkeypatch.setattr(
        vc,
        "cache_path",
        lambda sid: cache_root / f"{sid}.json",
    )
    return cache_root


def _cfg_with_cache(tmp_path, enabled: bool = True, max_messages: int = 20, ttl: float = 1800.0):
    """Return a minimal config dict with the cache flag set as requested."""
    return {
        "router_verdict_cache_enabled": enabled,
        "router_verdict_cache_max_messages": max_messages,
        "router_verdict_cache_ttl_secs": ttl,
    }


# ---------------------------------------------------------------------------
# 1. Flag OFF → build_fn called every time, no file written
# ---------------------------------------------------------------------------

def test_flag_off_calls_build_fn_every_time(tmp_cache_dir):
    from skill_hub.router.verdict_cache import get_or_build_stable_block

    calls = 0

    def build():
        nonlocal calls
        calls += 1
        return "block"

    cfg = _cfg_with_cache(None, enabled=False)

    get_or_build_stable_block(
        "session-1",
        current_domain_hints=["python"],
        current_plan_mode=False,
        current_msg_count=1,
        hard_switch=False,
        build_fn=build,
        cfg=cfg,
    )
    get_or_build_stable_block(
        "session-1",
        current_domain_hints=["python"],
        current_plan_mode=False,
        current_msg_count=2,
        hard_switch=False,
        build_fn=build,
        cfg=cfg,
    )

    assert calls == 2, "build_fn must be called on every turn when flag is off"
    assert not tmp_cache_dir.exists(), "no cache files must be written when flag is off"


def test_flag_off_no_file_for_empty_session(tmp_cache_dir):
    """Empty session_id also skips caching (even when flag is on)."""
    from skill_hub.router.verdict_cache import get_or_build_stable_block

    calls = 0

    def build():
        nonlocal calls
        calls += 1
        return "x"

    cfg = _cfg_with_cache(None, enabled=True)

    get_or_build_stable_block(
        "",
        current_domain_hints=["python"],
        current_plan_mode=False,
        current_msg_count=1,
        hard_switch=False,
        build_fn=build,
        cfg=cfg,
    )
    assert calls == 1
    assert not tmp_cache_dir.exists()


# ---------------------------------------------------------------------------
# 2. Flag ON, same session + stable domain + within limits → cache hit
# ---------------------------------------------------------------------------

def test_cache_hit_same_session(tmp_cache_dir):
    from skill_hub.router.verdict_cache import get_or_build_stable_block

    calls = 0

    def build():
        nonlocal calls
        calls += 1
        return "skills-block"

    cfg = _cfg_with_cache(None, enabled=True)

    result1 = get_or_build_stable_block(
        "sess-abc",
        current_domain_hints=["python"],
        current_plan_mode=False,
        current_msg_count=1,
        hard_switch=False,
        build_fn=build,
        cfg=cfg,
    )
    result2 = get_or_build_stable_block(
        "sess-abc",
        current_domain_hints=["python"],
        current_plan_mode=False,
        current_msg_count=2,
        hard_switch=False,
        build_fn=build,
        cfg=cfg,
    )

    assert calls == 1, "build_fn must be called exactly once; second call must use cache"
    assert result1 == result2 == "skills-block"


def test_cache_file_is_written(tmp_cache_dir):
    from skill_hub.router.verdict_cache import get_or_build_stable_block, cache_path

    cfg = _cfg_with_cache(None, enabled=True)

    get_or_build_stable_block(
        "sess-write",
        current_domain_hints=["api"],
        current_plan_mode=False,
        current_msg_count=0,
        hard_switch=False,
        build_fn=lambda: "my-block",
        cfg=cfg,
    )

    p = cache_path("sess-write")
    assert p.is_file()
    entry = json.loads(p.read_text())
    assert entry["stable_block"] == "my-block"
    assert entry["domain_hints"] == ["api"]


# ---------------------------------------------------------------------------
# 3. Invalidation cases
# ---------------------------------------------------------------------------

def test_invalidation_plan_mode_change(tmp_cache_dir):
    from skill_hub.router.verdict_cache import get_or_build_stable_block

    calls = 0

    def build():
        nonlocal calls
        calls += 1
        return f"block-{calls}"

    cfg = _cfg_with_cache(None, enabled=True)

    get_or_build_stable_block(
        "sess-pm",
        current_domain_hints=["python"],
        current_plan_mode=False,
        current_msg_count=1,
        hard_switch=False,
        build_fn=build,
        cfg=cfg,
    )
    get_or_build_stable_block(
        "sess-pm",
        current_domain_hints=["python"],
        current_plan_mode=True,  # changed
        current_msg_count=2,
        hard_switch=False,
        build_fn=build,
        cfg=cfg,
    )

    assert calls == 2, "plan_mode change must invalidate the cache"


def test_invalidation_hard_switch(tmp_cache_dir):
    from skill_hub.router.verdict_cache import get_or_build_stable_block

    calls = 0

    def build():
        nonlocal calls
        calls += 1
        return "b"

    cfg = _cfg_with_cache(None, enabled=True)

    get_or_build_stable_block(
        "sess-hs",
        current_domain_hints=["python"],
        current_plan_mode=False,
        current_msg_count=1,
        hard_switch=False,
        build_fn=build,
        cfg=cfg,
    )
    get_or_build_stable_block(
        "sess-hs",
        current_domain_hints=["python"],
        current_plan_mode=False,
        current_msg_count=2,
        hard_switch=True,  # enforcement applied
        build_fn=build,
        cfg=cfg,
    )

    assert calls == 2, "hard_switch=True must invalidate the cache"


def test_invalidation_new_domain(tmp_cache_dir):
    """A NEW domain entering the session triggers recompute (superset rule)."""
    from skill_hub.router.verdict_cache import get_or_build_stable_block

    calls = 0

    def build():
        nonlocal calls
        calls += 1
        return "b"

    cfg = _cfg_with_cache(None, enabled=True)

    get_or_build_stable_block(
        "sess-dom",
        current_domain_hints=["python"],
        current_plan_mode=False,
        current_msg_count=1,
        hard_switch=False,
        build_fn=build,
        cfg=cfg,
    )
    get_or_build_stable_block(
        "sess-dom",
        current_domain_hints=["python", "fastapi"],  # new domain added
        current_plan_mode=False,
        current_msg_count=2,
        hard_switch=False,
        build_fn=build,
        cfg=cfg,
    )

    assert calls == 2, "new domain entering the session must invalidate the cache"


def test_no_invalidation_domain_shrinks(tmp_cache_dir):
    """Current domain set is a strict subset of cached → reuse."""
    from skill_hub.router.verdict_cache import get_or_build_stable_block

    calls = 0

    def build():
        nonlocal calls
        calls += 1
        return "b"

    cfg = _cfg_with_cache(None, enabled=True)

    # Prime cache with two domains.
    get_or_build_stable_block(
        "sess-sub",
        current_domain_hints=["python", "fastapi"],
        current_plan_mode=False,
        current_msg_count=1,
        hard_switch=False,
        build_fn=build,
        cfg=cfg,
    )
    # Second call with subset of the cached domains — should reuse.
    get_or_build_stable_block(
        "sess-sub",
        current_domain_hints=["python"],
        current_plan_mode=False,
        current_msg_count=2,
        hard_switch=False,
        build_fn=build,
        cfg=cfg,
    )

    assert calls == 1, "domain subset of cached set must NOT invalidate the cache"


def test_invalidation_msg_count_exceeded(tmp_cache_dir):
    from skill_hub.router.verdict_cache import get_or_build_stable_block

    calls = 0

    def build():
        nonlocal calls
        calls += 1
        return "b"

    cfg = _cfg_with_cache(None, enabled=True, max_messages=5)

    get_or_build_stable_block(
        "sess-mc",
        current_domain_hints=["python"],
        current_plan_mode=False,
        current_msg_count=0,
        hard_switch=False,
        build_fn=build,
        cfg=cfg,
    )
    # Delta = 5 which equals max_messages=5 — threshold is `>= max_messages`
    get_or_build_stable_block(
        "sess-mc",
        current_domain_hints=["python"],
        current_plan_mode=False,
        current_msg_count=5,
        hard_switch=False,
        build_fn=build,
        cfg=cfg,
    )

    assert calls == 2, "msg_count delta >= max_messages must invalidate the cache"


def test_invalidation_ttl_expired(tmp_cache_dir, monkeypatch):
    from skill_hub.router import verdict_cache as vc

    calls = 0

    def build():
        nonlocal calls
        calls += 1
        return "b"

    cfg = _cfg_with_cache(None, enabled=True, ttl=10.0)

    # Write the first entry at t=0.
    fake_time = [1_000_000.0]
    monkeypatch.setattr(vc.time, "time", lambda: fake_time[0])

    vc.get_or_build_stable_block(
        "sess-ttl",
        current_domain_hints=["python"],
        current_plan_mode=False,
        current_msg_count=1,
        hard_switch=False,
        build_fn=build,
        cfg=cfg,
    )

    # Advance time past the TTL.
    fake_time[0] += 11.0

    vc.get_or_build_stable_block(
        "sess-ttl",
        current_domain_hints=["python"],
        current_plan_mode=False,
        current_msg_count=2,
        hard_switch=False,
        build_fn=build,
        cfg=cfg,
    )

    assert calls == 2, "TTL expiry must invalidate the cache"


# ---------------------------------------------------------------------------
# 4. Corrupt / missing cache file → fallback, never raises
# ---------------------------------------------------------------------------

def test_corrupt_cache_falls_back(tmp_cache_dir):
    from skill_hub.router.verdict_cache import get_or_build_stable_block, cache_path

    cfg = _cfg_with_cache(None, enabled=True)

    # Write a corrupt (non-JSON) file.
    p = cache_path("sess-bad")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{{not valid json}}")

    calls = 0

    def build():
        nonlocal calls
        calls += 1
        return "fresh"

    result = get_or_build_stable_block(
        "sess-bad",
        current_domain_hints=["python"],
        current_plan_mode=False,
        current_msg_count=1,
        hard_switch=False,
        build_fn=build,
        cfg=cfg,
    )

    assert result == "fresh"
    assert calls == 1, "corrupt cache must fall back to build_fn"


def test_missing_cache_file_falls_back(tmp_cache_dir):
    from skill_hub.router.verdict_cache import get_or_build_stable_block

    cfg = _cfg_with_cache(None, enabled=True)

    calls = 0

    def build():
        nonlocal calls
        calls += 1
        return "fresh"

    result = get_or_build_stable_block(
        "sess-missing",
        current_domain_hints=["python"],
        current_plan_mode=False,
        current_msg_count=1,
        hard_switch=False,
        build_fn=build,
        cfg=cfg,
    )

    assert result == "fresh"
    assert calls == 1


def test_cache_never_raises_on_write_error(tmp_cache_dir, monkeypatch):
    """Even if _save raises, get_or_build_stable_block returns build_fn result."""
    from skill_hub.router import verdict_cache as vc

    monkeypatch.setattr(vc, "_save", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full")))

    cfg = _cfg_with_cache(None, enabled=True)

    result = vc.get_or_build_stable_block(
        "sess-err",
        current_domain_hints=["python"],
        current_plan_mode=False,
        current_msg_count=1,
        hard_switch=False,
        build_fn=lambda: "safe",
        cfg=cfg,
    )

    assert result == "safe"


# ---------------------------------------------------------------------------
# 5. format_stable_block / format_volatile partition
# ---------------------------------------------------------------------------

def test_stable_block_contains_only_skills_and_plugins():
    """format_stable_block must contain skills/plugins and NOTHING else."""
    from skill_hub.router.verdict import format_stable_block

    v = _make_verdict(skills=["skill-a", "skill-b"], plugins=["plugin-x"])
    block = format_stable_block(v)

    assert "Preloaded skills" in block
    assert "Suggested plugins" in block
    # Must NOT contain the volatile header.
    assert "[Router] sonnet" not in block
    assert "confidence=" not in block


def test_stable_block_empty_when_no_skills():
    from skill_hub.router.verdict import format_stable_block

    v = _make_verdict(skills=[], plugins=[])
    assert format_stable_block(v) == ""


def test_volatile_contains_header():
    """format_volatile must contain the per-turn header line."""
    from skill_hub.router.verdict import format_volatile

    v = _make_verdict()
    vol = format_volatile(v)

    assert "[Router] sonnet" in vol
    assert "confidence=" in vol
    assert "tier=" in vol


def test_volatile_does_not_contain_skills():
    from skill_hub.router.verdict import format_volatile

    v = _make_verdict(skills=["skill-a"])
    vol = format_volatile(v)

    assert "Preloaded skills" not in vol


def test_concatenation_preserves_all_content():
    """stable + volatile together must contain everything format_system_message used to."""
    from skill_hub.router.verdict import (
        format_stable_block,
        format_volatile,
        format_system_message,
        Verdict,
    )
    from dataclasses import replace

    v = Verdict(
        model="haiku",
        confidence=0.75,
        reasoning="test reasoning",
        tier_used=2,
        preload_skills=["skill-x"],
        preload_plugins=["plugin-y"],
        compact_hint={"suggest_compact": True, "reason": "window full"},
        subtasks=["do thing A", "do thing B"],
        settings_opt={"key": "some_key", "value": True, "reason": "perf"},
    )

    full = format_system_message(v)
    stable = format_stable_block(v)
    volatile = format_volatile(v)

    # Everything in format_system_message must appear in stable+volatile.
    for line in full.splitlines():
        assert line in (stable + "\n" + volatile).splitlines(), (
            f"line missing from stable+volatile: {line!r}"
        )

    # No content is lost.
    assert "skill-x" in stable
    assert "plugin-y" in stable
    assert "haiku" in volatile
    assert "COMPACT ADVISORY" in volatile
    assert "do thing A" in volatile
    assert "some_key" in volatile


def test_format_system_message_stable_first():
    """The stable block must come BEFORE the volatile block in the combined output."""
    from skill_hub.router.verdict import format_system_message

    v = _make_verdict(skills=["alpha"])
    msg = format_system_message(v)
    lines = msg.splitlines()

    skill_idx = next(i for i, l in enumerate(lines) if "Preloaded skills" in l)
    header_idx = next(i for i, l in enumerate(lines) if "confidence=" in l)

    assert skill_idx < header_idx, (
        "Stable block (skills line) must appear BEFORE the volatile header — "
        "the reorder from issue #88 is intentional to form a cacheable prefix."
    )


def test_format_system_message_backward_compatible_no_skills():
    """When there are no skills the output is just the volatile header (no blank lines)."""
    from skill_hub.router.verdict import format_system_message

    v = _make_verdict(skills=[], plugins=[])
    msg = format_system_message(v)

    assert "confidence=" in msg
    assert not msg.startswith("\n")


def test_preloaded_skills_still_sorted():
    """Sorted order invariant must hold through the new split."""
    from skill_hub.router.verdict import format_stable_block

    v = _make_verdict(skills=["zeta", "alpha", "mu"])
    block = format_stable_block(v)

    assert "alpha, mu, zeta" in block
