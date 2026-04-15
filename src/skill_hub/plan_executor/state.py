"""Per-plan execution state — sidecar JSON next to the plan YAML.

Plan at ~/.claude/plans/foo.yaml → state at ~/.claude/plans/foo.state.json.
Holds step statuses and execution history so re-running skips done steps and
honors depends_on.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal


StepStatus = Literal["pending", "running", "done", "failed", "escalated"]


def state_path_for(plan_path: Path | str) -> Path:
    p = Path(plan_path)
    return p.with_suffix(".state.json")


def load_state(plan_path: Path | str) -> dict[str, Any]:
    sp = state_path_for(plan_path)
    if not sp.exists():
        return {"steps": {}}
    try:
        return json.loads(sp.read_text())
    except (OSError, json.JSONDecodeError):
        return {"steps": {}}


def save_state(plan_path: Path | str, state: dict[str, Any]) -> None:
    sp = state_path_for(plan_path)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(state, indent=2))


def step_status(state: dict[str, Any], step_id: str) -> StepStatus:
    return state.get("steps", {}).get(step_id, {}).get("status", "pending")


def mark_step(
    plan_path: Path | str,
    step_id: str,
    status: StepStatus,
    *,
    tier: str | None = None,
    model: str | None = None,
    acceptance_output: str | None = None,
    notes: str | None = None,
) -> None:
    state = load_state(plan_path)
    entry = state.setdefault("steps", {}).setdefault(step_id, {})
    entry["status"] = status
    entry["updated_at"] = time.time()
    if tier is not None:
        entry["tier"] = tier
    if model is not None:
        entry["model"] = model
    if acceptance_output is not None:
        entry["acceptance_output"] = acceptance_output[-2000:]
    if notes is not None:
        entry["notes"] = notes
    save_state(plan_path, state)


def deps_satisfied(state: dict[str, Any], depends_on: list[str]) -> list[str]:
    """Return the subset of ``depends_on`` that are NOT yet done."""
    return [d for d in depends_on if step_status(state, d) != "done"]
