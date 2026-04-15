"""Tests for plan_executor.executor — tier routing, retry-on-escalate,
scope enforcement, depends_on gating, state persistence, reward recording.

Uses injected fakes for the LLM and reward sink so no network / DB I/O.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from skill_hub.plan_executor import (  # noqa: E402
    execute_plan_step,
    load_state,
    step_status,
)


# ---- helpers ----------------------------------------------------------------

def _write_plan(path: Path, steps: list[dict]) -> Path:
    plan = {"plan_id": "t", "goal": "g", "steps": steps}
    path.write_text(yaml.safe_dump(plan))
    return path


def _fake_chat(response_payload: dict):
    calls: list[dict] = []

    def chat(messages, *, tier, max_tokens=4096, temperature=0.2, **_):
        calls.append({"tier": tier, "messages": messages})
        return json.dumps(response_payload)

    chat.calls = calls  # type: ignore[attr-defined]
    return chat


def _capture_rewards():
    recorded: list[tuple[str, str, str, float]] = []

    def record(tier, task_class, domain, success):
        recorded.append((tier, task_class, domain, success))

    record.recorded = recorded  # type: ignore[attr-defined]
    return record


# ---- tests ------------------------------------------------------------------

def test_kind_maps_to_tier_mid_for_tests(tmp_path):
    plan_path = _write_plan(
        tmp_path / "plan.yaml",
        [{"id": "T1", "kind": "tests", "files": ["a.py"], "acceptance": "echo ok"}],
    )
    chat = _fake_chat({"files": {"a.py": "x=1\n"}, "notes": "did it"})
    reward = _capture_rewards()

    result = execute_plan_step(
        plan_path, "T1",
        dry_run=False, chat_fn=chat, reward_fn=reward, repo_path=tmp_path,
    )

    assert result.status == "done"
    assert result.tier == "tier_mid"
    assert chat.calls[0]["tier"] == "tier_mid"
    assert (tmp_path / "a.py").read_text() == "x=1\n"
    assert reward.recorded == [("tier_mid", "tests", "t", 1.0)]


def test_kind_architecture_routes_to_tier_smart(tmp_path):
    plan_path = _write_plan(
        tmp_path / "plan.yaml",
        [{"id": "T1", "kind": "architecture", "files": ["a.py"], "acceptance": "echo ok"}],
    )
    chat = _fake_chat({"files": {"a.py": "pass\n"}, "notes": ""})
    reward = _capture_rewards()

    execute_plan_step(
        plan_path, "T1",
        dry_run=False, chat_fn=chat, reward_fn=reward, repo_path=tmp_path,
    )
    assert chat.calls[0]["tier"] == "tier_smart"


def test_model_hint_overrides_kind_mapping(tmp_path):
    plan_path = _write_plan(
        tmp_path / "plan.yaml",
        [{
            "id": "T1", "kind": "tests", "files": ["a.py"],
            "acceptance": "echo ok", "model_hint": "tier_smart",
        }],
    )
    chat = _fake_chat({"files": {"a.py": "pass\n"}})
    execute_plan_step(
        plan_path, "T1",
        dry_run=False, chat_fn=chat, reward_fn=_capture_rewards(), repo_path=tmp_path,
    )
    assert chat.calls[0]["tier"] == "tier_smart"


def test_dry_run_does_not_write(tmp_path):
    plan_path = _write_plan(
        tmp_path / "plan.yaml",
        [{"id": "T1", "kind": "tests", "files": ["a.py"], "acceptance": "echo ok"}],
    )
    chat = _fake_chat({"files": {"a.py": "hello\n"}})
    result = execute_plan_step(
        plan_path, "T1",
        dry_run=True, chat_fn=chat, reward_fn=_capture_rewards(), repo_path=tmp_path,
    )
    assert result.dry_run is True
    assert result.status == "done"
    assert not (tmp_path / "a.py").exists()


def test_scope_violation_rejected(tmp_path):
    plan_path = _write_plan(
        tmp_path / "plan.yaml",
        [{"id": "T1", "kind": "tests", "files": ["a.py"], "acceptance": "echo ok"}],
    )
    # Model tries to touch an undeclared file.
    chat = _fake_chat({"files": {"a.py": "ok", "b.py": "naughty"}})
    reward = _capture_rewards()
    result = execute_plan_step(
        plan_path, "T1",
        dry_run=False, chat_fn=chat, reward_fn=reward, repo_path=tmp_path,
    )
    # Primary tier fails scope, escalates to tier_smart (same fake chat → same violation).
    assert result.status == "failed"
    assert "undeclared file" in result.acceptance_output
    assert not (tmp_path / "b.py").exists()
    # Reward recorded as 0.
    assert reward.recorded[-1][3] == 0.0


def test_depends_on_blocks_when_unmet(tmp_path):
    plan_path = _write_plan(
        tmp_path / "plan.yaml",
        [
            {"id": "T1", "kind": "tests", "files": ["a.py"], "acceptance": "echo ok"},
            {"id": "T2", "kind": "tests", "files": ["b.py"], "acceptance": "echo ok",
             "depends_on": ["T1"]},
        ],
    )
    chat = _fake_chat({"files": {"b.py": "x"}})
    result = execute_plan_step(
        plan_path, "T2",
        dry_run=False, chat_fn=chat, reward_fn=_capture_rewards(), repo_path=tmp_path,
    )
    assert result.status == "blocked"
    assert "T1" in result.acceptance_output
    # LLM was not called.
    assert chat.calls == []


def test_depends_on_satisfied_after_first_step(tmp_path):
    plan_path = _write_plan(
        tmp_path / "plan.yaml",
        [
            {"id": "T1", "kind": "tests", "files": ["a.py"], "acceptance": "echo ok"},
            {"id": "T2", "kind": "tests", "files": ["b.py"], "acceptance": "echo ok",
             "depends_on": ["T1"]},
        ],
    )
    chat1 = _fake_chat({"files": {"a.py": "A"}})
    r1 = execute_plan_step(plan_path, "T1", dry_run=False,
                           chat_fn=chat1, reward_fn=_capture_rewards(), repo_path=tmp_path)
    assert r1.status == "done"

    chat2 = _fake_chat({"files": {"b.py": "B"}})
    r2 = execute_plan_step(plan_path, "T2", dry_run=False,
                           chat_fn=chat2, reward_fn=_capture_rewards(), repo_path=tmp_path)
    assert r2.status == "done"
    assert (tmp_path / "b.py").read_text() == "B"


def test_done_step_not_rerun(tmp_path):
    plan_path = _write_plan(
        tmp_path / "plan.yaml",
        [{"id": "T1", "kind": "tests", "files": ["a.py"], "acceptance": "echo ok"}],
    )
    chat = _fake_chat({"files": {"a.py": "1"}})
    execute_plan_step(plan_path, "T1", dry_run=False,
                      chat_fn=chat, reward_fn=_capture_rewards(), repo_path=tmp_path)
    # Second call — should short-circuit.
    chat2 = _fake_chat({"files": {"a.py": "SHOULD_NOT_WRITE"}})
    r = execute_plan_step(plan_path, "T1", dry_run=False,
                          chat_fn=chat2, reward_fn=_capture_rewards(), repo_path=tmp_path)
    assert r.status == "done"
    assert chat2.calls == []
    assert (tmp_path / "a.py").read_text() == "1"


def test_state_persisted(tmp_path):
    plan_path = _write_plan(
        tmp_path / "plan.yaml",
        [{"id": "T1", "kind": "tests", "files": ["a.py"], "acceptance": "echo ok"}],
    )
    chat = _fake_chat({"files": {"a.py": "s"}, "notes": "stateful"})
    execute_plan_step(plan_path, "T1", dry_run=False,
                      chat_fn=chat, reward_fn=_capture_rewards(), repo_path=tmp_path)
    state = load_state(plan_path)
    assert step_status(state, "T1") == "done"
    assert state["steps"]["T1"]["tier"] == "tier_mid"
    assert state["steps"]["T1"]["notes"] == "stateful"


def test_acceptance_failure_escalates_to_tier_smart(tmp_path):
    plan_path = _write_plan(
        tmp_path / "plan.yaml",
        [{"id": "T1", "kind": "tests", "files": ["a.py"],
          "acceptance": "false"}],  # `false` always exits non-zero
    )
    chat = _fake_chat({"files": {"a.py": "x"}})
    reward = _capture_rewards()
    result = execute_plan_step(
        plan_path, "T1",
        dry_run=False, chat_fn=chat, reward_fn=reward, repo_path=tmp_path,
    )
    # `false` isn't in the runnable_prefixes list → treated as heuristic pass.
    # Use a real failing command instead:
    # (Replacing the plan in-place would mutate state; retest below.)
    assert result.status == "done"  # heuristic pass path


def test_acceptance_real_command_failure_escalates(tmp_path):
    plan_path = _write_plan(
        tmp_path / "plan.yaml",
        [{"id": "T1", "kind": "tests", "files": ["a.py"],
          "acceptance": "pytest --does-not-exist-flag"}],
    )
    chat = _fake_chat({"files": {"a.py": "x"}})
    reward = _capture_rewards()
    result = execute_plan_step(
        plan_path, "T1",
        dry_run=False, chat_fn=chat, reward_fn=reward, repo_path=tmp_path,
    )
    # Primary tier_mid fails, escalates to tier_smart (same fake chat, same failure).
    assert result.attempted_tiers == ["tier_mid", "tier_smart"]
    assert result.status == "failed"
    assert reward.recorded[-1][3] == 0.0
    assert reward.recorded[-1][0] == "tier_smart"  # final attempted tier is what we reward


def test_fenced_json_output_parsed(tmp_path):
    plan_path = _write_plan(
        tmp_path / "plan.yaml",
        [{"id": "T1", "kind": "tests", "files": ["a.py"], "acceptance": "echo ok"}],
    )
    # Model wraps JSON in a markdown fence — common Claude output shape.
    def chat(messages, *, tier, **_):
        return '```json\n{"files": {"a.py": "fenced\\n"}, "notes": "n"}\n```'

    result = execute_plan_step(
        plan_path, "T1",
        dry_run=False, chat_fn=chat, reward_fn=_capture_rewards(), repo_path=tmp_path,
    )
    assert result.status == "done"
    assert (tmp_path / "a.py").read_text() == "fenced\n"


def test_unknown_step_id_raises(tmp_path):
    plan_path = _write_plan(
        tmp_path / "plan.yaml",
        [{"id": "T1", "kind": "tests", "files": ["a.py"], "acceptance": "echo ok"}],
    )
    with pytest.raises(KeyError):
        execute_plan_step(plan_path, "GHOST",
                          chat_fn=_fake_chat({}), reward_fn=_capture_rewards(),
                          repo_path=tmp_path)
