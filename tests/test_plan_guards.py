"""Tests for plan_executor.guards — default rules, user overlay, executor integration."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from skill_hub.plan_executor import (  # noqa: E402
    DEFAULT_GUARDS,
    GuardRule,
    check_post_dispatch,
    check_pre_dispatch,
    execute_plan_step,
    load_guards,
)


def _write_plan(path, steps):
    path.write_text(yaml.safe_dump({"plan_id": "t", "goal": "g", "steps": steps}))
    return path


def _fake_chat(payload):
    calls = []
    def chat(messages, *, tier, **_):
        calls.append({"tier": tier})
        return json.dumps(payload)
    chat.calls = calls
    return chat


def _reward_sink():
    recorded = []
    def r(tier, task_class, domain, success):
        recorded.append((tier, task_class, domain, success))
    r.recorded = recorded
    return r


# ---- guard rule matching ----------------------------------------------------

def test_default_guards_catch_claude_md(tmp_path):
    matches = check_pre_dispatch(["CLAUDE.md"], tmp_path, DEFAULT_GUARDS)
    assert len(matches) == 1
    assert matches[0].rule.name == "no-claude-context-commits"


def test_default_guards_catch_nested_claude_md(tmp_path):
    matches = check_pre_dispatch(["subdir/CLAUDE.md"], tmp_path, DEFAULT_GUARDS)
    assert any(m.rule.name == "no-claude-context-commits" for m in matches)


def test_default_guards_catch_env_files(tmp_path):
    matches = check_pre_dispatch(["config/.env.production"], tmp_path, DEFAULT_GUARDS)
    assert any(m.rule.name == "no-secret-files" for m in matches)


def test_default_guards_catch_migration_files(tmp_path):
    matches = check_pre_dispatch(
        ["db/migrations/v0042__add_tiles.sql"], tmp_path, DEFAULT_GUARDS
    )
    assert any(m.rule.name == "no-proactive-migrations" for m in matches)


def test_default_guards_allow_normal_files(tmp_path):
    matches = check_pre_dispatch(
        ["src/app/router.py", "tests/test_router.py"], tmp_path, DEFAULT_GUARDS
    )
    assert matches == []


def test_content_guards_catch_ai_attribution(tmp_path):
    changes = {"src/foo.py": "# Co-Authored-By: Claude\npass"}
    matches = check_post_dispatch(changes, tmp_path, DEFAULT_GUARDS)
    assert any(m.rule.name == "no-ai-attribution" for m in matches)


def test_content_guards_allow_clean_code(tmp_path):
    changes = {"src/foo.py": "def foo(): pass"}
    matches = check_post_dispatch(changes, tmp_path, DEFAULT_GUARDS)
    assert matches == []


def test_user_guards_merge_with_defaults(tmp_path):
    user = tmp_path / "plan_guards.yaml"
    user.write_text(yaml.safe_dump({
        "guards": [{
            "name": "my-danger-zone",
            "action": "force_tier_smart",
            "file_globs": ["src/dangerous/**"],
            "reason": "legacy module",
        }]
    }))
    guards = load_guards(user)
    names = {g.name for g in guards}
    assert "my-danger-zone" in names
    # Defaults still present.
    assert "no-ai-attribution" in names


def test_repo_glob_scoping(tmp_path):
    rule = GuardRule(
        name="repo-scoped",
        action="escalate",
        file_globs=["**/foo.py"],
        content_patterns=[],
        reason="only applies to /repo-a",
        repo_glob="*/repo-a",
    )
    # Wrong repo — no match.
    assert check_pre_dispatch(["foo.py"], Path("/tmp/repo-b"), [rule]) == []
    # Right repo — match.
    assert len(check_pre_dispatch(["foo.py"], Path("/tmp/repo-a"), [rule])) == 1


# ---- executor integration ---------------------------------------------------

def test_executor_escalates_on_guard_file_match(tmp_path):
    plan_path = _write_plan(
        tmp_path / "plan.yaml",
        [{"id": "T1", "kind": "tests", "files": ["CLAUDE.md"], "acceptance": "echo ok"}],
    )
    chat = _fake_chat({"files": {"CLAUDE.md": "x"}})
    reward = _reward_sink()

    result = execute_plan_step(
        plan_path, "T1", dry_run=False,
        chat_fn=chat, reward_fn=reward, repo_path=tmp_path,
    )
    assert result.status == "escalated"
    assert chat.calls == []   # LLM not called
    assert "no-claude-context-commits" in result.acceptance_output
    # Reward NOT recorded on pre-dispatch escalation (bandit shouldn't learn here).
    assert reward.recorded == []


def test_executor_escalates_on_ai_attribution_in_output(tmp_path):
    plan_path = _write_plan(
        tmp_path / "plan.yaml",
        [{"id": "T1", "kind": "tests", "files": ["a.py"], "acceptance": "echo ok"}],
    )
    # Model returns code containing a forbidden attribution.
    chat = _fake_chat({
        "files": {"a.py": "# Co-Authored-By: Claude\ndef x(): pass"},
        "notes": "implemented",
    })
    reward = _reward_sink()

    result = execute_plan_step(
        plan_path, "T1", dry_run=False,
        chat_fn=chat, reward_fn=reward, repo_path=tmp_path,
    )
    assert result.status == "escalated"
    assert not (tmp_path / "a.py").exists()  # file not written
    # Reward IS recorded here (0.0) because the LLM call was made.
    assert reward.recorded[-1][3] == 0.0


def test_force_tier_smart_guard_upgrades_routing(tmp_path):
    # Custom guard list bypassing defaults.
    custom = [GuardRule(
        name="upgrade-zone",
        action="force_tier_smart",
        file_globs=["src/critical/**"],
        content_patterns=[],
        reason="danger zone",
    )]
    plan_path = _write_plan(
        tmp_path / "plan.yaml",
        [{"id": "T1", "kind": "tests", "files": ["src/critical/x.py"], "acceptance": "echo ok"}],
    )
    chat = _fake_chat({"files": {"src/critical/x.py": "pass"}})
    reward = _reward_sink()

    result = execute_plan_step(
        plan_path, "T1", dry_run=False,
        chat_fn=chat, reward_fn=reward, repo_path=tmp_path, guards=custom,
    )
    # Step.kind=tests would normally route to tier_mid; guard forced tier_smart.
    assert chat.calls[0]["tier"] == "tier_smart"
    assert result.tier == "tier_smart"
    assert result.status == "done"


def test_guards_override_empty_allows_all(tmp_path):
    """Passing guards=[] disables all guards (useful for testing)."""
    plan_path = _write_plan(
        tmp_path / "plan.yaml",
        [{"id": "T1", "kind": "tests", "files": ["CLAUDE.md"], "acceptance": "echo ok"}],
    )
    chat = _fake_chat({"files": {"CLAUDE.md": "x"}})
    result = execute_plan_step(
        plan_path, "T1", dry_run=False,
        chat_fn=chat, reward_fn=_reward_sink(), repo_path=tmp_path, guards=[],
    )
    assert result.status == "done"


def test_user_guards_file_corrupt_falls_back_to_defaults(tmp_path):
    bad = tmp_path / "plan_guards.yaml"
    bad.write_text("not: valid: yaml: [")
    guards = load_guards(bad)
    # Bad YAML → fall back to defaults only.
    assert len(guards) == len(DEFAULT_GUARDS)
