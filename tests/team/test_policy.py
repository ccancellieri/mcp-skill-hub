"""Tests for src/skill_hub/team/policy.py — Component 2."""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC))

import pytest  # noqa: E402

from skill_hub.team import policy  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TASK_KINDS = ["review", "arch", "issues", "implement"]


# ---------------------------------------------------------------------------
# Default effort is xhigh
# ---------------------------------------------------------------------------


def test_default_effort_is_xhigh():
    plan = policy.resolve_team_plan("review")
    assert plan["effort"] == "xhigh"


def test_default_loops_is_three():
    plan = policy.resolve_team_plan("review")
    assert plan["loops"] == 3


# ---------------------------------------------------------------------------
# Substrate per task kind
# ---------------------------------------------------------------------------


def test_review_substrate_is_team():
    assert policy.resolve_team_plan("review")["substrate"] == "team"


def test_arch_substrate_is_team():
    assert policy.resolve_team_plan("arch")["substrate"] == "team"


def test_issues_substrate_is_workflow():
    assert policy.resolve_team_plan("issues")["substrate"] == "workflow"


def test_implement_substrate_is_workflow():
    assert policy.resolve_team_plan("implement")["substrate"] == "workflow"


# ---------------------------------------------------------------------------
# Roster sizes and content
# ---------------------------------------------------------------------------


def test_review_roster_size():
    # 4 reviewers + human_voice_writer + github_operator = 6
    plan = policy.resolve_team_plan("review")
    assert len(plan["roles"]) == 6


def test_review_has_four_reviewer_entries():
    plan = policy.resolve_team_plan("review")
    reviewers = [r for r in plan["roles"] if r["role"] == policy.reviewer]
    assert len(reviewers) == 4


def test_review_reviewer_lenses():
    plan = policy.resolve_team_plan("review")
    lenses = {r["lens"] for r in plan["roles"] if r["role"] == policy.reviewer}
    assert lenses == {"security", "correctness", "performance", "tests"}


def test_review_includes_human_voice_writer_and_github_operator():
    plan = policy.resolve_team_plan("review")
    roles = {r["role"] for r in plan["roles"]}
    assert policy.human_voice_writer in roles
    assert policy.github_operator in roles


def test_arch_roster_size():
    # 3 arch_analyst lenses + 1 devils-advocate + human_voice_writer = 5
    plan = policy.resolve_team_plan("arch")
    assert len(plan["roles"]) == 5


def test_arch_has_four_arch_analyst_entries():
    plan = policy.resolve_team_plan("arch")
    analysts = [r for r in plan["roles"] if r["role"] == policy.arch_analyst]
    assert len(analysts) == 4


def test_arch_lenses_include_devils_advocate():
    plan = policy.resolve_team_plan("arch")
    lenses = {r.get("lens") for r in plan["roles"] if r["role"] == policy.arch_analyst}
    assert "devils-advocate" in lenses
    assert "layer-1" in lenses
    assert "layer-2" in lenses
    assert "layer-3" in lenses


def test_issues_roster_stages():
    plan = policy.resolve_team_plan("issues")
    roles_ordered = [r["role"] for r in plan["roles"]]
    assert roles_ordered == [
        policy.github_operator,
        policy.arch_analyst,
        policy.human_voice_writer,
    ]


def test_issues_lenses():
    plan = policy.resolve_team_plan("issues")
    lenses = [r.get("lens") for r in plan["roles"]]
    assert lenses == ["fetch-triage", "classify-impact", "draft-updates"]


def test_implement_roster_stages():
    plan = policy.resolve_team_plan("implement")
    roles_ordered = [r["role"] for r in plan["roles"]]
    assert roles_ordered == [
        policy.arch_analyst,
        policy.code_implementer,
        policy.reviewer,
        policy.mechanical_refactorer,
        policy.human_voice_writer,
        policy.github_operator,
    ]


def test_implement_lenses():
    plan = policy.resolve_team_plan("implement")
    lenses = [r.get("lens") for r in plan["roles"]]
    assert lenses == ["design", "build", "verify", "clean", "pr-prose", "open-pr"]


# ---------------------------------------------------------------------------
# Agent names
# ---------------------------------------------------------------------------


def test_agent_names_are_correct():
    expected = {
        policy.arch_analyst:          "team-arch-analyst",
        policy.code_implementer:      "team-code-implementer",
        policy.mechanical_refactorer: "team-mechanical-refactorer",
        policy.github_operator:       "team-github-operator",
        policy.human_voice_writer:    "team-human-voice-writer",
        policy.reviewer:              "team-reviewer",
    }
    plan = policy.resolve_team_plan("review")
    for entry in plan["roles"]:
        assert entry["agent"] == expected[entry["role"]]


# ---------------------------------------------------------------------------
# Tier → cc_model mapping
# ---------------------------------------------------------------------------


def test_tier_cheap_maps_to_haiku():
    assert policy._TIER_TO_CC_MODEL["tier_cheap"] == "haiku"


def test_tier_mid_maps_to_haiku():
    assert policy._TIER_TO_CC_MODEL["tier_mid"] == "haiku"


def test_tier_smart_maps_to_sonnet():
    assert policy._TIER_TO_CC_MODEL["tier_smart"] == "sonnet"


def test_tier_planner_maps_to_opus():
    assert policy._TIER_TO_CC_MODEL["tier_planner"] == "opus"


# ---------------------------------------------------------------------------
# xhigh review — reviewers must be opus
# ---------------------------------------------------------------------------


def test_xhigh_review_reviewers_are_opus():
    plan = policy.resolve_team_plan("review", effort="xhigh")
    for entry in plan["roles"]:
        if entry["role"] == policy.reviewer:
            assert entry["cc_model"] == "opus", (
                f"Expected opus for reviewer at xhigh, got {entry['cc_model']}"
            )


# ---------------------------------------------------------------------------
# github_operator never exceeds haiku (at any effort)
# ---------------------------------------------------------------------------


def test_github_operator_never_exceeds_haiku():
    for effort in policy.EFFORT_LEVELS:
        tier = policy._TIER_FLOOR[policy.github_operator][effort]
        cc_model = policy._TIER_TO_CC_MODEL[tier]
        assert cc_model == "haiku", (
            f"github_operator at effort={effort} resolved to {cc_model}, expected haiku"
        )


# ---------------------------------------------------------------------------
# Invalid inputs raise ValueError
# ---------------------------------------------------------------------------


def test_invalid_task_kind_raises():
    with pytest.raises(ValueError, match="task_kind"):
        policy.resolve_team_plan("nonexistent")


def test_invalid_effort_raises():
    with pytest.raises(ValueError, match="effort"):
        policy.resolve_team_plan("review", effort="ultra")


def test_estimate_invalid_task_kind_raises():
    with pytest.raises(ValueError):
        policy.estimate_cost("bogus")


def test_estimate_invalid_effort_raises():
    with pytest.raises(ValueError):
        policy.estimate_cost("review", effort="max")


# ---------------------------------------------------------------------------
# Estimate monotonic non-decreasing in effort
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_kind", TASK_KINDS)
def test_estimate_agent_calls_monotonic(task_kind: str):
    prev_calls = -1
    for effort in policy.EFFORT_LEVELS:
        est = policy.estimate_cost(task_kind, effort)
        assert est["agent_calls"] >= prev_calls, (
            f"{task_kind} agent_calls not monotonic at effort={effort}: "
            f"{est['agent_calls']} < {prev_calls}"
        )
        prev_calls = est["agent_calls"]


@pytest.mark.parametrize("task_kind", TASK_KINDS)
def test_estimate_token_budget_high_monotonic(task_kind: str):
    prev_high = -1
    for effort in policy.EFFORT_LEVELS:
        est = policy.estimate_cost(task_kind, effort)
        assert est["token_budget_high"] >= prev_high, (
            f"{task_kind} token_budget_high not monotonic at effort={effort}: "
            f"{est['token_budget_high']} < {prev_high}"
        )
        prev_high = est["token_budget_high"]


# ---------------------------------------------------------------------------
# Estimate output has expected keys
# ---------------------------------------------------------------------------


def test_estimate_has_all_keys():
    est = policy.estimate_cost("review", "xhigh")
    for key in (
        "agent_calls",
        "token_budget_low",
        "token_budget_high",
        "rough_minutes_low",
        "rough_minutes_high",
        "assumptions",
    ):
        assert key in est, f"Missing key: {key}"


def test_estimate_assumptions_is_list_of_strings():
    est = policy.estimate_cost("implement", "high")
    assert isinstance(est["assumptions"], list)
    assert all(isinstance(a, str) for a in est["assumptions"])


# ---------------------------------------------------------------------------
# Loops per effort
# ---------------------------------------------------------------------------


def test_loops_per_effort():
    expected = {"low": 0, "medium": 1, "high": 2, "xhigh": 3}
    for effort, loops in expected.items():
        plan = policy.resolve_team_plan("review", effort=effort)
        assert plan["loops"] == loops, f"loops mismatch at effort={effort}"
