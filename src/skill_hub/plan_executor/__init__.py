"""Plan-YAML validation utility.

The in-process LLM execution engine (author/run/step + guards/scope/state/
walker/runner) was retired when orchestration converged on Claude Code's native
agent teams + Workflow tool. What remains is the pure, dependency-light schema
validator — handy for hand-authored plan YAML — plus the kind→tier table, which
now lives authoritatively in ``skill_hub.team.policy``.
"""

from ..team.policy import KIND_TIER_MAP as TIER_MAP
from .validator import (
    VALID_KINDS,
    VALID_TIERS,
    PlanValidationError,
    validate_plan,
    validate_plan_file,
)

__all__ = [
    "PlanValidationError",
    "TIER_MAP",
    "VALID_KINDS",
    "VALID_TIERS",
    "validate_plan",
    "validate_plan_file",
]
