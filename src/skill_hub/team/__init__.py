"""Specialized team orchestration layer — Phase 1.

Public API:
    from skill_hub.team import policy
    policy.resolve_team_plan(task_kind, effort)
    policy.estimate_cost(task_kind, effort)
"""
from . import policy
from .policy import (
    KIND_TIER_MAP,
    VALID_KINDS,
    VALID_ROLES,
    estimate_cost,
    resolve_team_plan,
    tier_for_kind,
)

__all__ = [
    "KIND_TIER_MAP",
    "VALID_KINDS",
    "VALID_ROLES",
    "estimate_cost",
    "policy",
    "resolve_team_plan",
    "tier_for_kind",
]
