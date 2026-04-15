"""Tests for session memory compaction (cookbook port).

Covers:
- Prompt template rendering (no LLM needed)
- Memory persistence: write + read round-trip
- Background build scheduling (mocked LLM)
- Incremental update path
- Transcript tail reader
- Build suppression when a build is already running
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import sys
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Fixtures

@pytest.fixture(autouse=True)
def _clean_builds():
    """Clear the active-builds registry before each test."""
    from skill_hub.router import session_memory as sm
    with sm._STATE_LOCK:
        sm._ACTIVE_BUILDS.clear()
    yield
    with sm._STATE_LOCK:
        sm._ACTIVE_BUILDS.clear()


@pytest.fixture()
def tmp_memory_dir(tmp_path, monkeypatch):
    """Redirect memory_dir() to a temp directory."""
    from skill_hub.router import session_memory as sm

    def _patched():
        return tmp_path / "session-memory"

    monkeypatch.setattr(sm, "memory_dir", _patched)
    monkeypatch.setattr(
        sm, "memory_path",
        lambda sid: _patched() / f"{sid}.md",
    )
    return tmp_path / "session-memory"


# ---------------------------------------------------------------------------
# Prompt rendering

def test_render_prompt_has_six_sections():
    from skill_hub.router.session_memory import _render_prompt

    result = _render_prompt(transcript="user: hello\nassistant: hi")
    sections = [
        "## User Intent",
        "## Completed Work",
        "## Errors & Corrections",
        "## Active Work",
        "## Pending Tasks",
        "## Key References",
    ]
    for s in sections:
        assert s in result, f"missing section: {s}"


def test_render_prompt_includes_transcript():
    from skill_hub.router.session_memory import _render_prompt

    result = _render_prompt(transcript="user: fix bug\nassistant: done")
    assert "fix bug" in result


def test_render_prompt_includes_previous_memory_when_present():
    from skill_hub.router.session_memory import _render_prompt

    result = _render_prompt(
        transcript="new turn",
        previous_memory="# User Intent\nOriginal goal",
    )
    assert "Original goal" in result
    assert "refresh" in result.lower() or "previous" in result.lower()


def test_render_prompt_omits_previous_block_when_empty():
    from skill_hub.router.session_memory import _render_prompt

    result = _render_prompt(transcript="only turn", previous_memory="")
    # Should not contain the "Previous session memory" header
    assert "Previous session memory" not in result


# ---------------------------------------------------------------------------
# Persistence round-trip

def test_write_read_roundtrip(tmp_memory_dir, monkeypatch):
    from skill_hub.router import session_memory as sm

    monkeypatch.setattr(sm, "memory_dir", lambda: tmp_memory_dir)
    monkeypatch.setattr(
        sm, "memory_path",
        lambda sid: tmp_memory_dir / f"{sid}.md",
    )

    sid = "test-session-abc"
    content = "## User Intent\nFix the bug.\n\n## Active Work\nIn progress."
    path = sm.write_memory(sid, content)

    assert path.exists()
    recovered = sm.read_memory(sid)
    assert recovered == content


def test_read_missing_memory_returns_empty(tmp_path, monkeypatch):
    from skill_hub.router import session_memory as sm

    monkeypatch.setattr(sm, "memory_dir", lambda: tmp_path / "mem")
    monkeypatch.setattr(
        sm, "memory_path",
        lambda sid: tmp_path / "mem" / f"{sid}.md",
    )

    assert sm.read_memory("no-such-session") == ""


# ---------------------------------------------------------------------------
# Background thread / schedule_build

def test_schedule_build_calls_provider(tmp_memory_dir, monkeypatch):
    """schedule_build launches a daemon thread that calls the LLM."""
    from skill_hub.router import session_memory as sm

    monkeypatch.setattr(sm, "memory_dir", lambda: tmp_memory_dir)
    monkeypatch.setattr(
        sm, "memory_path",
        lambda sid: tmp_memory_dir / f"{sid}.md",
    )
    monkeypatch.setattr(sm, "_is_enabled", lambda: True)

    fake_memory = (
        "## User Intent\nFix it.\n\n## Completed Work\nDone.\n\n"
        "## Errors & Corrections\nNone.\n\n## Active Work\nIdle.\n\n"
        "## Pending Tasks\nNone.\n\n## Key References\nN/A."
    )

    provider_mock = MagicMock()
    provider_mock.complete.return_value = fake_memory

    with patch("skill_hub.router.session_memory.get_provider", return_value=provider_mock):
        outcome = sm.schedule_build(
            "sess-1",
            lambda: "user: hi\nassistant: hello",
            incremental=False,
        )
        assert outcome.scheduled
        sm.wait_for_active_builds(timeout=5.0)

    result = sm.read_memory("sess-1")
    assert "## User Intent" in result
    provider_mock.complete.assert_called_once()


def test_schedule_build_suppressed_when_already_running(monkeypatch):
    """A second schedule_build while one is running returns scheduled=False."""
    from skill_hub.router import session_memory as sm

    monkeypatch.setattr(sm, "_is_enabled", lambda: True)

    barrier = threading.Barrier(2)
    released = threading.Event()

    def slow_provider():
        barrier.wait(timeout=3)
        released.wait(timeout=3)
        return "user: hi"

    with patch("skill_hub.router.session_memory.get_provider"):
        with patch.object(sm, "_run_build", side_effect=lambda *a, **k: None):
            # Manually insert a fake alive thread
            fake_thread = threading.Thread(target=lambda: time.sleep(10), daemon=True)
            fake_thread.start()
            with sm._STATE_LOCK:
                sm._ACTIVE_BUILDS["sess-dup"] = fake_thread

            outcome = sm.schedule_build("sess-dup", lambda: "text")
            assert not outcome.scheduled
            assert "already running" in outcome.reason

            fake_thread.join(timeout=0)  # don't wait — just cleanup tracking


def test_schedule_build_disabled_when_config_off(monkeypatch):
    from skill_hub.router import session_memory as sm

    monkeypatch.setattr(sm, "_is_enabled", lambda: False)
    outcome = sm.schedule_build("sess-off", lambda: "text")
    assert not outcome.scheduled
    assert "disabled" in outcome.reason


# ---------------------------------------------------------------------------
# Incremental update

def test_incremental_update_calls_update_fn(tmp_memory_dir, monkeypatch):
    from skill_hub.router import session_memory as sm

    monkeypatch.setattr(sm, "memory_dir", lambda: tmp_memory_dir)
    monkeypatch.setattr(
        sm, "memory_path",
        lambda sid: tmp_memory_dir / f"{sid}.md",
    )
    monkeypatch.setattr(sm, "_is_enabled", lambda: True)

    prev = (
        "## User Intent\nOld goal.\n\n## Completed Work\nOld work.\n\n"
        "## Errors & Corrections\nNone.\n\n## Active Work\nOld state.\n\n"
        "## Pending Tasks\nOld pending.\n\n## Key References\nOld refs."
    )
    sm.write_memory("sess-inc", prev)

    new_memory = prev.replace("Old goal", "New goal")
    provider_mock = MagicMock()
    provider_mock.complete.return_value = new_memory

    with patch("skill_hub.router.session_memory.get_provider", return_value=provider_mock):
        result = sm.update_session_memory(prev, "user: more work\nassistant: done")

    assert "New goal" in result
    # Prompt must contain the previous memory in context
    call_args = provider_mock.complete.call_args
    assert "Old goal" in call_args[0][0] or "Old goal" in str(call_args)


# ---------------------------------------------------------------------------
# Transcript tail reader

def test_read_transcript_tail_full_file(tmp_path):
    from skill_hub.router.session_memory import read_transcript_tail

    f = tmp_path / "t.jsonl"
    content = '{"role":"user","content":"hello"}\n' * 50
    f.write_text(content)

    result = read_transcript_tail(f)
    assert "hello" in result
    assert len(result) == len(content)


def test_read_transcript_tail_respects_cap(tmp_path):
    from skill_hub.router.session_memory import read_transcript_tail

    f = tmp_path / "t.jsonl"
    line = '{"role":"user","content":"x"}\n'
    f.write_text(line * 1000)

    result = read_transcript_tail(f, max_bytes=200)
    assert len(result) <= 200 + len(line)  # +1 line for partial skip


def test_read_transcript_tail_missing_returns_empty():
    from skill_hub.router.session_memory import read_transcript_tail

    assert read_transcript_tail("/nonexistent/file.jsonl") == ""


# ---------------------------------------------------------------------------
# Config defaults

def test_config_keys_present():
    from skill_hub import config as cfg

    defaults = cfg._DEFAULTS
    assert defaults["session_memory_enabled"] is True
    assert defaults["session_memory_min_messages"] == 6
    assert defaults["session_memory_tier"] == "tier_mid"
    assert defaults["session_memory_inject_on_resume"] is True
    assert defaults["session_memory_max_transcript_bytes"] == 200_000
    assert defaults["session_memory_inject_max_chars"] == 8000
