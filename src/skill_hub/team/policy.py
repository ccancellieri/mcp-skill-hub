"""Model·effort policy for the specialized team orchestration layer.

Pure-Python, no I/O, no LLM calls — fully unit-testable.

Component 2 of the Phase-1 team orchestration design.
"""
from __future__ import annotations

import math
from typing import Any

# ---------------------------------------------------------------------------
# Effort levels
# ---------------------------------------------------------------------------
EFFORT_LEVELS: list[str] = ["low", "medium", "high", "xhigh"]
DEFAULT_EFFORT: str = "xhigh"

# ---------------------------------------------------------------------------
# Role constants
# ---------------------------------------------------------------------------
arch_analyst = "arch_analyst"
code_implementer = "code_implementer"
mechanical_refactorer = "mechanical_refactorer"
github_operator = "github_operator"
human_voice_writer = "human_voice_writer"
reviewer = "reviewer"

VALID_ROLES: tuple[str, ...] = (
    arch_analyst,
    code_implementer,
    mechanical_refactorer,
    github_operator,
    human_voice_writer,
    reviewer,
)

# ---------------------------------------------------------------------------
# Role × effort → tier floor
# Row = role, column = effort (low, medium, high, xhigh)
# ---------------------------------------------------------------------------
_TIER_FLOOR: dict[str, dict[str, str]] = {
    arch_analyst:         {"low": "tier_mid",   "medium": "tier_smart", "high": "tier_smart", "xhigh": "tier_planner"},
    reviewer:             {"low": "tier_mid",   "medium": "tier_mid",   "high": "tier_smart", "xhigh": "tier_planner"},
    human_voice_writer:   {"low": "tier_smart", "medium": "tier_smart", "high": "tier_planner", "xhigh": "tier_planner"},
    code_implementer:     {"low": "tier_mid",   "medium": "tier_mid",   "high": "tier_smart", "xhigh": "tier_smart"},
    mechanical_refactorer:{"low": "tier_cheap", "medium": "tier_mid",   "high": "tier_mid",   "xhigh": "tier_smart"},
    github_operator:      {"low": "tier_cheap", "medium": "tier_mid",   "high": "tier_mid",   "xhigh": "tier_mid"},
}

# ---------------------------------------------------------------------------
# Tier → Claude Code model alias
# tier_cheap / tier_mid → "haiku"
# tier_smart            → "sonnet"
# tier_planner          → "opus"
# ---------------------------------------------------------------------------
_TIER_TO_CC_MODEL: dict[str, str] = {
    "tier_cheap":   "haiku",
    "tier_mid":     "haiku",
    "tier_smart":   "sonnet",
    "tier_planner": "opus",
}

# Literal fallback model IDs (mirrors config defaults; read from config when available)
_TIER_TO_MODEL_ID_FALLBACK: dict[str, str] = {
    "tier_cheap":   "ollama/qwen2.5-coder:3b",
    "tier_mid":     "anthropic/claude-haiku-4-5",
    "tier_smart":   "anthropic/claude-sonnet-4-6",
    "tier_planner": "anthropic/claude-opus-4-6",
}


def _get_tier_to_model_id() -> dict[str, str]:
    """Return tier → model_id map by reading config, falling back to literals."""
    try:
        from skill_hub.config import get_config  # type: ignore[import]

        cfg = get_config()
        providers: dict[str, str] = cfg.get("llm_providers", {})
        result = dict(_TIER_TO_MODEL_ID_FALLBACK)
        for tier_key in ("tier_cheap", "tier_mid", "tier_smart", "tier_planner"):
            if tier_key in providers:
                result[tier_key] = providers[tier_key]
        return result
    except Exception:  # pragma: no cover — import may fail in isolated test envs
        return dict(_TIER_TO_MODEL_ID_FALLBACK)


# ---------------------------------------------------------------------------
# Verification loops per effort
# ---------------------------------------------------------------------------
_LOOPS: dict[str, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "xhigh": 3,
}

# ---------------------------------------------------------------------------
# Agent name per role
# ---------------------------------------------------------------------------
_AGENT_NAME: dict[str, str] = {
    arch_analyst:          "team-arch-analyst",
    code_implementer:      "team-code-implementer",
    mechanical_refactorer: "team-mechanical-refactorer",
    github_operator:       "team-github-operator",
    human_voice_writer:    "team-human-voice-writer",
    reviewer:              "team-reviewer",
}

# ---------------------------------------------------------------------------
# Tools hint per role (matches Component 1 agent definitions)
# ---------------------------------------------------------------------------
_TOOLS_HINT: dict[str, str] = {
    arch_analyst:          "Read, Grep, Glob, WebFetch",
    code_implementer:      "Read, Edit, Write, Bash, Grep, Glob",
    mechanical_refactorer: "Read, Edit, Grep, Glob, Bash",
    github_operator:       "Bash, Read, Grep",
    human_voice_writer:    "Read, WebFetch",
    reviewer:              "Read, Grep, Glob, Bash",
}

# ---------------------------------------------------------------------------
# Task rosters
# Each entry: (role, lens_or_None)
# ---------------------------------------------------------------------------
_TASK_ROSTERS: dict[str, tuple[str, list[tuple[str, str | None]]]] = {
    "review": (
        "team",
        [
            (reviewer,           "security"),
            (reviewer,           "correctness"),
            (reviewer,           "performance"),
            (reviewer,           "tests"),
            (human_voice_writer, None),
            (github_operator,    None),
        ],
    ),
    "arch": (
        "team",
        [
            (arch_analyst,       "layer-1"),
            (arch_analyst,       "layer-2"),
            (arch_analyst,       "layer-3"),
            (arch_analyst,       "devils-advocate"),
            (human_voice_writer, None),
        ],
    ),
    "issues": (
        "workflow",
        [
            (github_operator,    "fetch-triage"),
            (arch_analyst,       "classify-impact"),
            (human_voice_writer, "draft-updates"),
        ],
    ),
    "implement": (
        "workflow",
        [
            (arch_analyst,         "design"),
            (code_implementer,     "build"),
            (reviewer,             "verify"),
            (mechanical_refactorer,"clean"),
            (human_voice_writer,   "pr-prose"),
            (github_operator,      "open-pr"),
        ],
    ),
}

_VALID_TASK_KINDS: tuple[str, ...] = tuple(_TASK_ROSTERS.keys())


def _build_role_entry(
    role: str,
    lens: str | None,
    effort: str,
    tier_to_model_id: dict[str, str],
) -> dict[str, Any]:
    tier = _TIER_FLOOR[role][effort]
    entry: dict[str, Any] = {
        "role": role,
        "tier": tier,
        "cc_model": _TIER_TO_CC_MODEL[tier],
        "model_id": tier_to_model_id.get(tier, _TIER_TO_MODEL_ID_FALLBACK[tier]),
        "agent": _AGENT_NAME[role],
        "tools_hint": _TOOLS_HINT[role],
    }
    if lens is not None:
        entry["lens"] = lens
    # Re-order so lens appears after role when present (cosmetic)
    ordered: dict[str, Any] = {"role": entry.pop("role")}
    if "lens" in entry:
        ordered["lens"] = entry.pop("lens")
    ordered.update(entry)
    return ordered


def resolve_team_plan(
    task_kind: str,
    effort: str = DEFAULT_EFFORT,
) -> dict[str, Any]:
    """Return a fully-resolved team plan for the given task kind and effort.

    Raises ValueError on invalid task_kind or effort.
    """
    if task_kind not in _TASK_ROSTERS:
        raise ValueError(
            f"Invalid task_kind {task_kind!r}. Valid values: "
            + ", ".join(_VALID_TASK_KINDS)
        )
    if effort not in _LOOPS:
        raise ValueError(
            f"Invalid effort {effort!r}. Valid values: " + ", ".join(EFFORT_LEVELS)
        )

    substrate, roster = _TASK_ROSTERS[task_kind]
    loops = _LOOPS[effort]
    tier_to_model_id = _get_tier_to_model_id()

    roles = [
        _build_role_entry(role, lens, effort, tier_to_model_id)
        for role, lens in roster
    ]

    return {
        "task_kind": task_kind,
        "effort": effort,
        "substrate": substrate,
        "loops": loops,
        "roles": roles,
    }


# ---------------------------------------------------------------------------
# Per-model base output-token bands (heuristic, ±50%)
# ---------------------------------------------------------------------------
_TOKEN_BANDS: dict[str, tuple[int, int]] = {
    "opus":   (80_000, 160_000),
    "sonnet": (50_000, 110_000),
    "haiku":  (15_000,  40_000),
}

# Assumption strings surfaced in the estimate
_ASSUMPTIONS: list[str] = [
    "per-role base output-token bands: opus 80k–160k, sonnet 50k–110k, haiku 15k–40k (±50%)",
    "verified roles (reviewer) are called (1 + loops) times; all other roles once",
    "rough_minutes: low = ceil(agent_calls × 1.5), high = agent_calls × 4",
    "estimates are heuristic projections, actual costs may vary significantly",
]


def estimate_cost(
    task_kind: str,
    effort: str = DEFAULT_EFFORT,
) -> dict[str, Any]:
    """Return a heuristic cost estimate for the given task kind and effort.

    Monotonic non-decreasing in effort for the same task_kind on
    agent_calls and token_budget_high.

    Raises ValueError on invalid inputs.
    """
    plan = resolve_team_plan(task_kind, effort)
    loops = plan["loops"]

    token_low = 0
    token_high = 0
    agent_calls = 0

    for role_entry in plan["roles"]:
        role_name = role_entry["role"]
        cc_model = role_entry["cc_model"]
        band_low, band_high = _TOKEN_BANDS[cc_model]

        # reviewer roles are "verified" — called (1 + loops) times
        call_count = (1 + loops) if role_name == reviewer else 1
        agent_calls += call_count
        token_low += band_low * call_count
        token_high += band_high * call_count

    rough_minutes_low = math.ceil(agent_calls * 1.5)
    rough_minutes_high = agent_calls * 4

    return {
        "agent_calls": agent_calls,
        "token_budget_low": token_low,
        "token_budget_high": token_high,
        "rough_minutes_low": rough_minutes_low,
        "rough_minutes_high": rough_minutes_high,
        "assumptions": list(_ASSUMPTIONS),
    }
