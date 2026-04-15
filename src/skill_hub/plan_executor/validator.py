"""Plan YAML validator — schema, kind enum, file-path existence, dependency graph.

Kept framework-free: pure stdlib + pyyaml. Returns a list of error strings so
the authoring loop (claude -p / SDK / in-session) can feed them back for fixing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


VALID_KINDS: frozenset[str] = frozenset(
    {"architecture", "integration", "boilerplate", "tests", "docs"}
)

TIER_MAP: dict[str, str] = {
    "architecture": "tier_smart",
    "integration": "tier_smart",
    "boilerplate": "tier_mid",
    "tests": "tier_mid",
    "docs": "tier_mid",
}

VALID_TIERS: frozenset[str] = frozenset({"tier_cheap", "tier_mid", "tier_smart"})


class PlanValidationError(ValueError):
    """Raised when a plan fails validation. ``errors`` holds the full list."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


def _check_top_level(plan: Any) -> list[str]:
    errs: list[str] = []
    if not isinstance(plan, dict):
        return [f"plan must be a mapping, got {type(plan).__name__}"]
    if not isinstance(plan.get("plan_id"), str) or not plan["plan_id"].strip():
        errs.append("plan_id: required non-empty string")
    if not isinstance(plan.get("goal"), str) or not plan["goal"].strip():
        errs.append("goal: required non-empty string")
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        errs.append("steps: required non-empty list")
    if "invariants" in plan and not isinstance(plan["invariants"], list):
        errs.append("invariants: must be a list of strings if present")
    return errs


def _check_step(step: Any, idx: int, seen_ids: set[str]) -> tuple[list[str], str | None]:
    prefix = f"steps[{idx}]"
    if not isinstance(step, dict):
        return [f"{prefix}: must be a mapping"], None

    errs: list[str] = []
    sid = step.get("id")
    if not isinstance(sid, str) or not sid.strip():
        errs.append(f"{prefix}.id: required non-empty string")
        sid = None
    elif sid in seen_ids:
        errs.append(f"{prefix}.id: duplicate id '{sid}'")
    else:
        seen_ids.add(sid)

    kind = step.get("kind")
    if kind not in VALID_KINDS:
        errs.append(
            f"{prefix}.kind: must be one of {sorted(VALID_KINDS)}, got {kind!r}"
        )

    files = step.get("files")
    if not isinstance(files, list) or not files:
        errs.append(f"{prefix}.files: required non-empty list")
    elif not all(isinstance(f, str) and f.strip() for f in files):
        errs.append(f"{prefix}.files: all entries must be non-empty strings")

    acceptance = step.get("acceptance")
    if not isinstance(acceptance, str) or not acceptance.strip():
        errs.append(f"{prefix}.acceptance: required non-empty string")

    for opt_list in ("protocols_ref", "pattern_ref", "depends_on"):
        if opt_list in step:
            val = step[opt_list]
            if not isinstance(val, list) or not all(
                isinstance(x, str) and x.strip() for x in val
            ):
                errs.append(f"{prefix}.{opt_list}: must be a list of non-empty strings")

    if "model_hint" in step and step["model_hint"] not in VALID_TIERS:
        errs.append(
            f"{prefix}.model_hint: must be one of {sorted(VALID_TIERS)}, "
            f"got {step['model_hint']!r}"
        )

    return errs, sid


def _check_depends_graph(plan: dict[str, Any]) -> list[str]:
    """Verify depends_on references point to known step ids and no cycles exist."""
    steps = plan.get("steps") or []
    ids = {s.get("id") for s in steps if isinstance(s, dict) and isinstance(s.get("id"), str)}
    errs: list[str] = []
    graph: dict[str, list[str]] = {}
    for s in steps:
        if not isinstance(s, dict):
            continue
        sid = s.get("id")
        if not isinstance(sid, str):
            continue
        deps = s.get("depends_on", []) or []
        if not isinstance(deps, list):
            continue
        for d in deps:
            if d not in ids:
                errs.append(f"steps[{sid}].depends_on: unknown id '{d}'")
        graph[sid] = [d for d in deps if d in ids]

    # Cycle check (DFS).
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {sid: WHITE for sid in graph}

    def dfs(node: str, stack: list[str]) -> None:
        color[node] = GRAY
        for nxt in graph.get(node, []):
            if color.get(nxt) == GRAY:
                cycle = stack[stack.index(nxt):] + [nxt] if nxt in stack else [node, nxt]
                errs.append(f"depends_on cycle: {' -> '.join(cycle)}")
                return
            if color.get(nxt) == WHITE:
                dfs(nxt, stack + [nxt])
        color[node] = BLACK

    for sid in list(graph.keys()):
        if color[sid] == WHITE:
            dfs(sid, [sid])

    return errs


def _check_file_existence(
    plan: dict[str, Any], repo_path: Path, *, strict: bool
) -> list[str]:
    """Check that protocols_ref and pattern_ref paths exist on disk.

    ``files`` is intentionally *not* checked: those are the paths a step
    will create or modify, so they may not exist yet. ``protocols_ref`` and
    ``pattern_ref`` are read-only context inputs and MUST exist.

    ``strict=False`` downgrades missing paths to no-op (used when the plan
    is authored before the repo layout is final).
    """
    if not strict:
        return []
    errs: list[str] = []
    for s in plan.get("steps") or []:
        if not isinstance(s, dict):
            continue
        sid = s.get("id", "?")
        for key in ("protocols_ref", "pattern_ref"):
            for rel in s.get(key, []) or []:
                if not isinstance(rel, str):
                    continue
                p = repo_path / rel
                if not p.exists():
                    errs.append(f"steps[{sid}].{key}: file not found: {rel}")
    return errs


def validate_plan(
    plan: dict[str, Any] | str,
    *,
    repo_path: Path | str | None = None,
    check_files: bool = True,
) -> dict[str, Any]:
    """Validate a plan given as a dict or a YAML string.

    Args:
        plan: The plan as a dict, or a YAML string to parse.
        repo_path: Root for resolving protocols_ref / pattern_ref. Defaults to cwd.
        check_files: If True, verify protocols_ref/pattern_ref exist on disk.

    Returns:
        The parsed plan dict (useful when input was a YAML string).

    Raises:
        PlanValidationError: with the full list of error messages.
    """
    if isinstance(plan, str):
        try:
            parsed = yaml.safe_load(plan)
        except yaml.YAMLError as e:
            raise PlanValidationError([f"YAML parse error: {e}"]) from e
    else:
        parsed = plan

    errs = _check_top_level(parsed)
    if errs:
        raise PlanValidationError(errs)

    seen: set[str] = set()
    for idx, step in enumerate(parsed.get("steps", [])):
        step_errs, _ = _check_step(step, idx, seen)
        errs.extend(step_errs)

    if not errs:
        errs.extend(_check_depends_graph(parsed))

    if not errs and check_files:
        root = Path(repo_path) if repo_path else Path.cwd()
        errs.extend(_check_file_existence(parsed, root, strict=True))

    if errs:
        raise PlanValidationError(errs)

    return parsed


def validate_plan_file(
    path: Path | str,
    *,
    repo_path: Path | str | None = None,
    check_files: bool = True,
) -> dict[str, Any]:
    """Load a YAML plan file from disk and validate it."""
    p = Path(path)
    if not p.exists():
        raise PlanValidationError([f"plan file not found: {p}"])
    try:
        text = p.read_text()
    except OSError as e:
        raise PlanValidationError([f"cannot read plan file: {e}"]) from e
    return validate_plan(text, repo_path=repo_path, check_files=check_files)
