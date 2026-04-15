"""Plan-aware executor — consumes a structured YAML plan authored by a
planning model (Opus) and dispatches each step to the right executor
model (Sonnet/Haiku) with a file-scoped context bundle.

Step 1 (this module currently): schema + validator.
Later steps add execute_plan_step, transparent plan authoring, teaching seeds.
"""

from .author import AuthorResult, author_plan
from .executor import StepResult, execute_plan_step
from .guards import (
    DEFAULT_GUARDS,
    GuardMatch,
    GuardRule,
    check_post_dispatch,
    check_pre_dispatch,
    load_guards,
)
from .runner import RunnerFailed, RunnerResult, RunnerUnavailable, resolve_runner
from .scope import ContextBundle, build_bundle
from .walker import RunResult, run_plan
from .state import load_state, mark_step, state_path_for, step_status
from .validator import (
    TIER_MAP,
    VALID_KINDS,
    PlanValidationError,
    validate_plan,
    validate_plan_file,
)

__all__ = [
    "AuthorResult",
    "ContextBundle",
    "DEFAULT_GUARDS",
    "GuardMatch",
    "GuardRule",
    "PlanValidationError",
    "RunnerFailed",
    "RunnerResult",
    "RunnerUnavailable",
    "RunResult",
    "StepResult",
    "TIER_MAP",
    "VALID_KINDS",
    "author_plan",
    "build_bundle",
    "check_post_dispatch",
    "check_pre_dispatch",
    "load_guards",
    "execute_plan_step",
    "load_state",
    "mark_step",
    "resolve_runner",
    "run_plan",
    "state_path_for",
    "step_status",
    "validate_plan",
    "validate_plan_file",
]
