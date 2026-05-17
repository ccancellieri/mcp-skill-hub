"""Unit tests for skill_hub.fanout.directive."""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC))

from skill_hub.fanout.directive import DispatchSpec, render_directive


def test_render_directive_empty():
    out = render_directive([], "G1")
    assert "G1" in out
    assert "no tasks created" in out


def test_render_directive_shapes_agent_calls():
    specs = [
        DispatchSpec(description="d1", prompt="p1",
                     worktree_path="/tmp/a", branch="cc/a", task_id=1),
        DispatchSpec(description="d2", prompt='quotes " ok', branch="cc/b",
                     worktree_path="/tmp/b", task_id=2),
    ]
    out = render_directive(specs, "GROUP")
    assert "group `GROUP`" in out
    assert out.count("Agent({") == 2
    assert "isolation: \"worktree\"" in out
    assert "task #1" in out and "task #2" in out
    # Quoted prompt content should be JSON-escaped (no naked unescaped quote).
    assert '"quotes \\" ok"' in out


def test_render_directive_includes_rollup_pointer():
    spec = DispatchSpec(description="d", prompt="p",
                        worktree_path="/tmp/x", branch="cc/x", task_id=5)
    out = render_directive([spec], "ABCD")
    assert "fanout_status" in out
    assert "fanout_close" in out
    assert "ABCD" in out


def test_directive_contains_no_ai_tooling_paths():
    """Global CLAUDE.md rule: directive prose must not leak ~/.claude/ paths."""
    spec = DispatchSpec(description="d", prompt="p",
                        worktree_path="/tmp/x", branch="cc/x", task_id=5)
    out = render_directive([spec], "Z")
    # ~/.claude/ should not appear; .claude/worktrees/ paths come from worktree_path
    # which is the user's own (not an AI plan/session/agent id).
    assert "~/.claude" not in out
    assert ".claude/plans" not in out
    assert ".claude/projects" not in out
