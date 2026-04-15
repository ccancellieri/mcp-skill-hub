"""run_plan — walk every step in dependency order, stop on first failure.

Uses a topological sort over ``depends_on`` to choose execution order.
Steps already marked ``done`` in the sidecar state are skipped (idempotent
resume after a partial run).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .executor import StepResult, execute_plan_step
from .guards import GuardRule
from .state import load_state, step_status
from .validator import validate_plan_file


@dataclass
class RunResult:
    plan_id: str
    total_steps: int
    results: list[StepResult] = field(default_factory=list)
    stopped_reason: str | None = None  # None when all steps completed

    @property
    def completed(self) -> int:
        return sum(1 for r in self.results if r.status == "done")

    @property
    def escalated(self) -> int:
        return sum(1 for r in self.results if r.status == "escalated")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == "failed")

    def as_markdown(self) -> str:
        lines = [
            f"# Plan {self.plan_id} — run summary",
            f"  steps: {self.completed}/{self.total_steps} done "
            f"(failed={self.failed}, escalated={self.escalated})",
        ]
        if self.stopped_reason:
            lines.append(f"  stopped: {self.stopped_reason}")
        lines.append("")
        for r in self.results:
            lines.append(r.as_markdown())
            lines.append("")
        return "\n".join(lines).rstrip()


def _topo_order(steps: list[dict]) -> list[str]:
    """Kahn's algorithm — preserves original ordering where deps allow."""
    id_to_step = {s["id"]: s for s in steps}
    indegree = {sid: 0 for sid in id_to_step}
    graph: dict[str, list[str]] = {sid: [] for sid in id_to_step}
    for s in steps:
        for dep in s.get("depends_on", []) or []:
            if dep in id_to_step:
                graph[dep].append(s["id"])
                indegree[s["id"]] += 1
    # Seed queue in original plan order (preserves author intent for parallel-safe steps).
    queue = [s["id"] for s in steps if indegree[s["id"]] == 0]
    out: list[str] = []
    while queue:
        sid = queue.pop(0)
        out.append(sid)
        for nxt in graph[sid]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    if len(out) != len(id_to_step):
        # Validator should have caught cycles, but be defensive.
        raise ValueError("depends_on cycle detected in plan")
    return out


def run_plan(
    plan_path: Path | str,
    *,
    dry_run: bool = True,
    repo_path: Path | str | None = None,
    chat_fn: Callable[..., str] | None = None,
    reward_fn: Callable[..., None] | None = None,
    guards: list[GuardRule] | None = None,
    stop_on_failure: bool = True,
) -> RunResult:
    """Execute every not-yet-done step of a plan in topological order.

    Args:
        plan_path: Path to the plan YAML.
        dry_run: Forward to each ``execute_plan_step`` call.
        repo_path: Root for file resolution. Defaults to cwd.
        chat_fn / reward_fn / guards: Injection points (tests + custom seeds).
        stop_on_failure: Halt on first ``failed``/``escalated`` outcome (default
            True — matches the "retry once on Sonnet then hand back" policy).

    Returns:
        RunResult with per-step outcomes and a stop reason if halted early.
    """
    plan_path = Path(plan_path).expanduser()
    repo = Path(repo_path).expanduser() if repo_path else Path.cwd()
    plan = validate_plan_file(plan_path, repo_path=repo, check_files=False)
    order = _topo_order(plan["steps"])

    out = RunResult(plan_id=plan["plan_id"], total_steps=len(order))

    for sid in order:
        state = load_state(plan_path)
        if step_status(state, sid) == "done":
            # Already complete from a prior partial run — skip silently.
            continue
        result = execute_plan_step(
            plan_path, sid,
            dry_run=dry_run,
            chat_fn=chat_fn,
            reward_fn=reward_fn,
            repo_path=repo,
            guards=guards,
        )
        out.results.append(result)

        if result.status == "blocked":
            # Dependency unmet — means a prior step failed earlier without
            # marking this one escalated. Stop cleanly.
            out.stopped_reason = f"step {sid} blocked: {result.acceptance_output}"
            return out
        if stop_on_failure and result.status in ("failed", "escalated"):
            out.stopped_reason = f"step {sid} {result.status} — halting per stop_on_failure"
            return out

    return out
