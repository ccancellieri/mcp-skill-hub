"""Tests for plan_executor.walker.run_plan — topological ordering, resume,
stop-on-failure, dry_run propagation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from skill_hub.plan_executor import load_state, run_plan, step_status  # noqa: E402


def _write_plan(path, steps):
    path.write_text(yaml.safe_dump({"plan_id": "w", "goal": "g", "steps": steps}))
    return path


def _chat_per_step(payloads_by_step_id):
    calls = []
    order = []

    def chat(messages, *, tier, **_):
        calls.append({"tier": tier})
        text = messages[-1]["content"]
        for sid, payload in payloads_by_step_id.items():
            if f"STEP ID: {sid}" in text:
                order.append(sid)
                return json.dumps(payload)
        raise AssertionError(f"no payload matched step in prompt: {text[:200]}")

    chat.calls = calls
    chat.order = order
    return chat


def _reward_sink():
    recorded = []
    def r(*args): recorded.append(args)
    r.recorded = recorded
    return r


def test_run_plan_executes_in_topological_order(tmp_path):
    plan_path = _write_plan(tmp_path / "p.yaml", [
        {"id": "C", "kind": "tests", "files": ["c.py"], "acceptance": "echo", "depends_on": ["B"]},
        {"id": "A", "kind": "tests", "files": ["a.py"], "acceptance": "echo"},
        {"id": "B", "kind": "tests", "files": ["b.py"], "acceptance": "echo", "depends_on": ["A"]},
    ])
    chat = _chat_per_step({
        "A": {"files": {"a.py": "1"}},
        "B": {"files": {"b.py": "2"}},
        "C": {"files": {"c.py": "3"}},
    })
    result = run_plan(plan_path, dry_run=False, chat_fn=chat,
                     reward_fn=_reward_sink(), repo_path=tmp_path, guards=[])
    assert chat.order == ["A", "B", "C"]
    assert result.completed == 3
    assert result.stopped_reason is None


def test_run_plan_skips_already_done(tmp_path):
    plan_path = _write_plan(tmp_path / "p.yaml", [
        {"id": "A", "kind": "tests", "files": ["a.py"], "acceptance": "echo"},
        {"id": "B", "kind": "tests", "files": ["b.py"], "acceptance": "echo", "depends_on": ["A"]},
    ])
    # First run completes A and B.
    chat1 = _chat_per_step({
        "A": {"files": {"a.py": "A"}},
        "B": {"files": {"b.py": "B"}},
    })
    run_plan(plan_path, dry_run=False, chat_fn=chat1,
             reward_fn=_reward_sink(), repo_path=tmp_path, guards=[])
    assert step_status(load_state(plan_path), "A") == "done"

    # Second run — both should skip (no LLM calls).
    chat2 = _chat_per_step({})  # would fail if called
    result = run_plan(plan_path, dry_run=False, chat_fn=chat2,
                      reward_fn=_reward_sink(), repo_path=tmp_path, guards=[])
    assert chat2.calls == []
    assert result.total_steps == 2
    assert result.completed == 0  # none executed this run
    assert result.stopped_reason is None


def test_run_plan_halts_on_failure(tmp_path):
    plan_path = _write_plan(tmp_path / "p.yaml", [
        {"id": "A", "kind": "tests", "files": ["a.py"], "acceptance": "echo"},
        {"id": "B", "kind": "tests", "files": ["b.py"],
         "acceptance": "pytest --bogus-flag-that-fails",
         "depends_on": ["A"]},
        {"id": "C", "kind": "tests", "files": ["c.py"], "acceptance": "echo", "depends_on": ["B"]},
    ])
    chat = _chat_per_step({
        "A": {"files": {"a.py": "A"}},
        "B": {"files": {"b.py": "B"}},
        "C": {"files": {"c.py": "C"}},  # should never be called
    })
    result = run_plan(plan_path, dry_run=False, chat_fn=chat,
                      reward_fn=_reward_sink(), repo_path=tmp_path, guards=[])
    assert result.completed == 1  # only A
    assert result.failed == 1     # B
    assert "C" not in chat.order
    assert "halting" in result.stopped_reason


def test_run_plan_halts_on_escalation(tmp_path):
    # Use a default-guard trigger: writing CLAUDE.md escalates.
    plan_path = _write_plan(tmp_path / "p.yaml", [
        {"id": "A", "kind": "tests", "files": ["a.py"], "acceptance": "echo"},
        {"id": "B", "kind": "tests", "files": ["CLAUDE.md"], "acceptance": "echo",
         "depends_on": ["A"]},
    ])
    chat = _chat_per_step({
        "A": {"files": {"a.py": "A"}},
        "B": {"files": {"CLAUDE.md": "x"}},
    })
    result = run_plan(plan_path, dry_run=False, chat_fn=chat,
                      reward_fn=_reward_sink(), repo_path=tmp_path)
    # Default guards active (no guards=[] override).
    assert result.escalated == 1
    assert "escalated" in result.stopped_reason


def test_run_plan_dry_run_propagates(tmp_path):
    plan_path = _write_plan(tmp_path / "p.yaml", [
        {"id": "A", "kind": "tests", "files": ["a.py"], "acceptance": "echo"},
    ])
    chat = _chat_per_step({"A": {"files": {"a.py": "preview"}}})
    result = run_plan(plan_path, dry_run=True, chat_fn=chat,
                      reward_fn=_reward_sink(), repo_path=tmp_path, guards=[])
    assert result.completed == 1
    assert not (tmp_path / "a.py").exists()


def test_run_plan_parallel_safe_steps_preserve_author_order(tmp_path):
    # A and B have no deps — should run in plan order A, B.
    plan_path = _write_plan(tmp_path / "p.yaml", [
        {"id": "A", "kind": "tests", "files": ["a.py"], "acceptance": "echo"},
        {"id": "B", "kind": "tests", "files": ["b.py"], "acceptance": "echo"},
    ])
    chat = _chat_per_step({
        "A": {"files": {"a.py": "A"}},
        "B": {"files": {"b.py": "B"}},
    })
    run_plan(plan_path, dry_run=False, chat_fn=chat,
             reward_fn=_reward_sink(), repo_path=tmp_path, guards=[])
    assert chat.order == ["A", "B"]
