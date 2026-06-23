"""#56: deterministic-first swap of the precompact + conversation_digest hot-paths.

By default these paths must condense recent input *extractively* (Kompress,
falling back to lossless deterministic compaction) WITHOUT invoking the local
LLM. The legacy abstractive local-LLM paths must remain reachable:
- precompact via the ``precompact_use_llm`` flag, and
- conversation_digest when ``eviction_enabled`` (it needs the profile-switch
  inference) or when forced via ``digest_use_llm``.

These tests monkeypatch the LLM entry points (``compact`` / ``conversation_digest``)
to a tripwire so we can assert the local LLM is NOT called on the default path,
and patch the extractive helper to a deterministic shrink so no model is loaded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture()
def isolated_cfg(tmp_path, monkeypatch):
    """Redirect CONFIG_PATH to a tmp file so cfg.set never touches the real config."""
    from skill_hub import config as cfg

    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "config.json")
    return cfg


class _FakeStore:
    """Minimal stand-in so the digest helpers never touch the real DB."""

    def save_conversation_state(self, **kwargs):  # noqa: D401 - stub
        return None

    def close(self):
        return None


def _patch_extractive_halves(monkeypatch):
    """Make kompress_prose deterministically shrink to half length (no model load)."""
    monkeypatch.setattr(
        "skill_hub.compression.kompress_prose",
        lambda text, **kw: text[: max(1, len(text) // 2)],
    )


# ---------------------------------------------------------------------------
# precompact
# ---------------------------------------------------------------------------

def test_precompact_deterministic_by_default(isolated_cfg, monkeypatch):
    """Default precompact condenses extractively and never calls the local LLM."""
    from skill_hub import cli

    called = {"llm": False}

    def _tripwire(*a, **k):
        called["llm"] = True
        return {"summary": "should not be used"}

    monkeypatch.setattr(cli, "compact", _tripwire)
    _patch_extractive_halves(monkeypatch)

    long_msg = "alpha beta gamma delta epsilon. " * 200  # well over the 1500 threshold
    hint = cli._precompact_hint(long_msg)

    assert called["llm"] is False, "deterministic precompact must not invoke the LLM"
    assert hint is not None
    assert hint.startswith("[Skill Hub — condensed view")
    assert len(hint) < len(long_msg)


def test_precompact_short_message_returns_none(isolated_cfg, monkeypatch):
    """Messages under the threshold are left untouched (no hint, no LLM)."""
    from skill_hub import cli

    monkeypatch.setattr(cli, "compact", lambda *a, **k: pytest.fail("LLM called"))
    assert cli._precompact_hint("tiny message") is None


def test_precompact_uses_llm_when_flagged(isolated_cfg, monkeypatch):
    """precompact_use_llm=True restores the legacy abstractive digest."""
    from skill_hub import cli

    isolated_cfg.set("precompact_use_llm", True)
    monkeypatch.setattr(cli, "should_run_llm", lambda *a, **k: True)
    monkeypatch.setattr(
        cli,
        "compact",
        lambda *a, **k: {"summary": "did things", "decisions": ["d1"], "open_questions": []},
    )

    hint = cli._precompact_hint("x " * 2000)
    assert hint is not None
    assert "pre-compacted summary" in hint
    assert "did things" in hint


# ---------------------------------------------------------------------------
# conversation_digest
# ---------------------------------------------------------------------------

def test_digest_extractive_by_default(isolated_cfg, monkeypatch):
    """Default digest (eviction off, digest_use_llm off) is extractive, no LLM."""
    from skill_hub import cli

    cli._session_messages.clear()
    called = {"llm": False}

    def _tripwire(*a, **k):
        called["llm"] = True
        return {}

    monkeypatch.setattr(cli, "conversation_digest", _tripwire)
    monkeypatch.setattr(cli, "SkillStore", _FakeStore)
    monkeypatch.setattr(cli, "_get_session_id", lambda: "sess-det")
    _patch_extractive_halves(monkeypatch)

    out = None
    for i in range(5):  # digest fires on the 5th message (every N=5)
        out = cli._conversation_digest_if_due(f"message {i} with enough words to condense " * 8)

    assert called["llm"] is False, "default digest must not invoke the LLM"
    assert out is not None
    assert "conversation digest" in out


def test_digest_uses_llm_when_eviction_on(isolated_cfg, monkeypatch):
    """eviction_enabled=True routes the digest through the abstractive LLM path."""
    from skill_hub import cli

    isolated_cfg.set("eviction_enabled", True)
    cli._session_messages.clear()
    called = {"llm": False}

    def _fake_cd(*a, **k):
        called["llm"] = True
        return {
            "current_focus": "the focus",
            "recent_decisions": ["d1"],
            "stale_topics": [],
            "suggested_profile": None,
        }

    monkeypatch.setattr(cli, "should_run_llm", lambda *a, **k: True)
    monkeypatch.setattr(cli, "ollama_available", lambda *a, **k: True)
    monkeypatch.setattr(cli, "conversation_digest", _fake_cd)
    monkeypatch.setattr(cli, "SkillStore", _FakeStore)
    monkeypatch.setattr(cli, "_get_session_id", lambda: "sess-llm")

    out = None
    for i in range(5):
        out = cli._conversation_digest_if_due(f"msg {i}")

    assert called["llm"] is True, "eviction path must invoke the abstractive digest"
    assert out is not None
    assert "the focus" in out


def test_digest_uses_llm_when_forced(isolated_cfg, monkeypatch):
    """digest_use_llm=True forces the abstractive digest even with eviction off."""
    from skill_hub import cli

    isolated_cfg.set("digest_use_llm", True)
    cli._session_messages.clear()
    called = {"llm": False}

    def _fake_cd(*a, **k):
        called["llm"] = True
        return {"current_focus": "f", "recent_decisions": [], "stale_topics": []}

    monkeypatch.setattr(cli, "should_run_llm", lambda *a, **k: True)
    monkeypatch.setattr(cli, "ollama_available", lambda *a, **k: True)
    monkeypatch.setattr(cli, "conversation_digest", _fake_cd)
    monkeypatch.setattr(cli, "SkillStore", _FakeStore)
    monkeypatch.setattr(cli, "_get_session_id", lambda: "sess-forced")

    for i in range(5):
        cli._conversation_digest_if_due(f"msg {i}")

    assert called["llm"] is True
