"""Deterministic agent-I/O compression guidance (#138)."""
from __future__ import annotations

from skill_hub.compression import agent_io_guidance


def test_dispatch_signal_returns_guidance():
    for msg in (
        "fan out agents on worktrees and implement",
        "dispatch a subagent to explore the codebase",
        "run these in parallel",
        "delegate the search to an explore agent",
        "orchestrate the migration across files",
    ):
        hint = agent_io_guidance(msg)
        assert hint is not None, msg
        assert "compressed findings" in hint


def test_ordinary_prompt_returns_none():
    for msg in (
        "fix the failing test in store.py",
        "what does this function return?",
        "add a column to the config defaults",
        "",
    ):
        assert agent_io_guidance(msg) is None, msg


def test_word_boundary_not_substring():
    # "agentic" contains "agent" as a substring but the word boundary must not match.
    assert agent_io_guidance("build an agentic coding assistant") is None
