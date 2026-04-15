"""author_plan — orchestrates: resolve runner → call it → validate output →
feed errors back for up to 2 fix-retries → write validated YAML to disk.

Transparent to the caller: they just get a path (or a directive string, when
we're in-session and the caller itself must do the authoring).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .runner import (
    RunnerFailed,
    RunnerName,
    RunnerResult,
    resolve_runner,
)
from .validator import PlanValidationError, validate_plan


DEFAULT_PLAN_DIR = Path.home() / ".claude" / "plans"
MAX_VALIDATION_RETRIES = 2


@dataclass
class AuthorResult:
    runner: RunnerName
    plan_path: Path | None       # None when a directive was returned
    directive: str | None        # set for in_session; caller must display
    validation_attempts: int
    used_api_tokens: bool        # True iff the API fallback ran

    def as_markdown(self) -> str:
        if self.directive:
            return (
                f"Authored via {self.runner} (directive returned — agent must "
                f"write the YAML itself):\n\n{self.directive}"
            )
        warn = "  ⚠ used API tokens\n" if self.used_api_tokens else ""
        return (
            f"Authored via {self.runner} → {self.plan_path}\n"
            f"{warn}"
            f"  validation attempts: {self.validation_attempts}"
        )


def _slugify(goal: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", goal.strip().lower()).strip("-")
    return s[:50] or "plan"


def _load_schema_hint(schema_path: Path | None) -> str:
    if schema_path is None:
        pkg_root = Path(__file__).resolve().parents[3]
        schema_path = pkg_root / "examples" / "plan.schema.yaml"
    if not schema_path.exists():
        # Inline minimal hint so authoring works even if schema file is missing.
        return (
            "plan_id: <slug>\n"
            "goal: <one-liner>\n"
            "steps:\n"
            "  - id: T1\n"
            "    kind: architecture|integration|boilerplate|tests|docs\n"
            "    files: [path/to/target.py]\n"
            "    protocols_ref: [path/to/protocols.py]  # optional\n"
            "    pattern_ref: [path/to/pattern.py]      # optional\n"
            "    acceptance: pytest -x tests/...\n"
            "    depends_on: [T0]                        # optional\n"
            "    model_hint: tier_smart                  # optional override\n"
        )
    return schema_path.read_text()


def author_plan(
    goal: str,
    repo_path: Path | str | None = None,
    *,
    plan_dir: Path | str | None = None,
    schema_path: Path | str | None = None,
    preferred_runner: RunnerName | None = None,
    chat_fn: Callable[..., str] | None = None,
) -> AuthorResult:
    """Author a plan YAML for ``goal`` targeting ``repo_path``.

    Resolution: HUB_PLAN_RUNNER env var > ``preferred_runner`` arg > auto-chain
    (in_session → cli → sdk → api). Writes validated YAML to
    ``<plan_dir>/<slug>.yaml``. Retries validation up to 2 times by feeding
    errors back to the same runner.

    Returns an ``AuthorResult``. For the in_session runner, ``plan_path`` is
    None and ``directive`` holds instructions for the calling agent.
    """
    repo = Path(repo_path).expanduser() if repo_path else Path.cwd()
    plan_dir_p = Path(plan_dir).expanduser() if plan_dir else DEFAULT_PLAN_DIR
    plan_dir_p.mkdir(parents=True, exist_ok=True)
    plan_path = plan_dir_p / f"{_slugify(goal)}.yaml"

    schema_hint = _load_schema_hint(
        Path(schema_path).expanduser() if schema_path else None
    )

    result: RunnerResult = resolve_runner(
        goal, repo, schema_hint, plan_path,
        preferred=preferred_runner, chat_fn=chat_fn,
    )

    # In-session path: the calling Claude Code agent will author the YAML
    # itself using the returned directive. We don't validate here — the
    # agent is expected to call validate_plan after writing.
    if result.yaml_text is None:
        return AuthorResult(
            runner=result.runner,
            plan_path=None,
            directive=result.directive,
            validation_attempts=0,
            used_api_tokens=False,
        )

    # Validate with fix-retries.
    attempts = 0
    yaml_text = result.yaml_text
    last_errors: list[str] = []
    while attempts <= MAX_VALIDATION_RETRIES:
        attempts += 1
        try:
            validate_plan(yaml_text, repo_path=repo, check_files=False)
            break
        except PlanValidationError as e:
            last_errors = e.errors
            if attempts > MAX_VALIDATION_RETRIES:
                raise RunnerFailed(
                    f"validation failed after {attempts} attempts via "
                    f"{result.runner}: {'; '.join(last_errors)}"
                ) from e
            # Ask the same runner to fix the errors.
            fix_goal = (
                f"FIX these plan validation errors and re-emit the YAML:\n"
                + "\n".join(f"  - {err}" for err in last_errors)
                + f"\n\nPrevious YAML:\n{yaml_text}"
            )
            fix_result = resolve_runner(
                fix_goal, repo, schema_hint, plan_path,
                preferred=result.runner,  # stick with the same runner
                chat_fn=chat_fn,
            )
            if fix_result.yaml_text is None:
                # Shouldn't happen — in_session already returned above.
                raise RunnerFailed(
                    f"fix-retry runner returned directive instead of YAML"
                )
            yaml_text = fix_result.yaml_text

    plan_path.write_text(yaml_text)
    return AuthorResult(
        runner=result.runner,
        plan_path=plan_path,
        directive=None,
        validation_attempts=attempts,
        used_api_tokens=(result.runner == "api"),
    )
