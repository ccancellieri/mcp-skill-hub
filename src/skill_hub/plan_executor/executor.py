"""execute_plan_step — the core dispatcher.

Flow:
  1. Load plan YAML, resolve step by id.
  2. Check depends_on via sidecar state.
  3. (Future: teaching-based escalation — hook point reserved.)
  4. Map kind → tier (honor model_hint override).
  5. Build file-scoped context bundle.
  6. Call LLM with a strict JSON-output contract.
  7. Apply returned file contents (unless dry_run).
  8. Run acceptance command; on failure, retry once on tier_smart.
  9. Record bandit reward; update state.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

from .guards import (
    GuardRule,
    check_post_dispatch,
    check_pre_dispatch,
    first_escalation,
    load_guards,
)
from .scope import build_bundle, ContextBundle
from .state import (
    deps_satisfied,
    load_state,
    mark_step,
    step_status,
)
from .validator import TIER_MAP, validate_plan_file


SYSTEM_PROMPT = """You are an execution agent for one atomic step of a larger plan.
You have been given:
  - The step's goal and acceptance criterion.
  - The content of the files in scope (target files + read-only references).
  - Hard invariants that must hold.

Your job: return a JSON object with the new content for each target file.

OUTPUT FORMAT — STRICT:
Return ONLY a JSON object (no markdown fence, no prose) with this shape:
{
  "files": {
    "relative/path/to/file.py": "<full new file content>",
    ...
  },
  "notes": "<one-line summary of what you did>"
}

Only include files you actually modified. The "files" map MUST be a subset of
the declared target files. Do NOT touch files outside the target list.
"""


@dataclass
class StepResult:
    step_id: str
    status: str                  # "done" | "failed" | "escalated" | "blocked"
    tier: str
    attempted_tiers: list[str]
    files_changed: list[str]
    model_notes: str
    acceptance_passed: bool
    acceptance_output: str
    bandit_reward: float
    dry_run: bool

    def as_markdown(self) -> str:
        status_emoji = {
            "done": "✓",
            "failed": "✗",
            "escalated": "↑",
            "blocked": "⏸",
        }.get(self.status, "?")
        lines = [
            f"{status_emoji} Step {self.step_id} — {self.status.upper()}",
            f"  tier: {self.tier} (attempted: {', '.join(self.attempted_tiers)})",
            f"  files: {', '.join(self.files_changed) if self.files_changed else '(none)'}",
            f"  acceptance: {'PASS' if self.acceptance_passed else 'FAIL'}",
            f"  reward: {self.bandit_reward:.2f}",
            f"  dry_run: {self.dry_run}",
        ]
        if self.model_notes:
            lines.append(f"  notes: {self.model_notes}")
        if not self.acceptance_passed and self.acceptance_output:
            lines.append(f"  output: {self.acceptance_output[-300:]}")
        return "\n".join(lines)


def _find_step(plan: dict[str, Any], step_id: str) -> dict[str, Any] | None:
    for s in plan.get("steps", []):
        if isinstance(s, dict) and s.get("id") == step_id:
            return s
    return None


def _pick_tier(step: dict[str, Any]) -> str:
    hint = step.get("model_hint")
    if hint:
        return hint
    return TIER_MAP.get(step.get("kind", ""), "tier_mid")


def _build_user_prompt(
    plan: dict[str, Any],
    step: dict[str, Any],
    bundle: ContextBundle,
) -> str:
    invariants = plan.get("invariants") or []
    inv_block = (
        "\n".join(f"  - {x}" for x in invariants) if invariants else "  (none)"
    )
    return (
        f"PLAN GOAL: {plan['goal']}\n"
        f"STEP ID: {step['id']}\n"
        f"STEP KIND: {step['kind']}\n"
        f"ACCEPTANCE: {step['acceptance']}\n\n"
        f"INVARIANTS:\n{inv_block}\n\n"
        f"TARGET FILES (you may only modify these):\n  - "
        + "\n  - ".join(step["files"])
        + f"\n\nCONTEXT:\n\n{bundle.render()}"
    )


_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def _parse_model_output(text: str) -> dict[str, Any]:
    """Tolerant JSON extractor — strips accidental markdown fencing."""
    cleaned = _JSON_FENCE.sub("", text).strip()
    # Fallback: grab the first { ... } block.
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _apply_changes(
    repo_path: Path,
    declared_files: set[str],
    changes: dict[str, str],
) -> list[str]:
    """Write each file from ``changes`` to disk. Reject any path not in
    ``declared_files`` (the step's target list) — enforces the scope contract.
    """
    written: list[str] = []
    for rel, content in changes.items():
        if rel not in declared_files:
            raise ValueError(
                f"model tried to modify undeclared file: {rel!r} "
                f"(allowed: {sorted(declared_files)})"
            )
        if not isinstance(content, str):
            raise ValueError(f"file {rel!r}: content must be a string")
        full = repo_path / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)
        written.append(rel)
    return written


def _run_acceptance(command: str, repo_path: Path) -> tuple[bool, str]:
    """Run the acceptance criterion. If it looks like a shell command
    (starts with a known tool name), run it. Otherwise treat as a heuristic
    label and return True (caller must interpret).
    """
    # Heuristic: if the acceptance string contains shell metacharacters or
    # starts with a common tool name, run it. Otherwise skip.
    cmdline = command.strip()
    runnable_prefixes = ("pytest", "pyright", "mypy", "ruff", "python", "sh", "bash", "npm", "yarn", "make")
    if not any(cmdline.split(" ", 1)[0] == p for p in runnable_prefixes):
        return True, f"(non-executable acceptance — assumed pass: {cmdline!r})"
    try:
        proc = subprocess.run(
            cmdline,
            shell=True,
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return False, "acceptance timed out after 300s"
    except OSError as e:
        return False, f"acceptance could not run: {e}"
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, output


# Type alias for the LLM callable — simplifies testing (inject a fake).
ChatFn = Callable[..., str]


def execute_plan_step(
    plan_path: Path | str,
    step_id: str,
    *,
    dry_run: bool = True,
    chat_fn: ChatFn | None = None,
    reward_fn: Callable[[str, str, str, float], None] | None = None,
    repo_path: Path | str | None = None,
    guards: list[GuardRule] | None = None,
) -> StepResult:
    """Execute one step from a validated plan YAML.

    Args:
        plan_path: Path to the plan YAML.
        step_id: Which step to run.
        dry_run: If True, do not write files to disk — just return the preview.
        chat_fn: Callable with signature ``(messages, *, tier, max_tokens, ...) -> str``.
                 Defaults to ``get_provider().chat`` via litellm. Injectable for tests.
        reward_fn: Callable ``(tier, task_class, domain, success) -> None`` called
                   after the step completes. Defaults to skill_hub.router.bandit.record_reward
                   against the default SkillStore.
        repo_path: Root for resolving files. Defaults to cwd.

    Returns:
        StepResult. The sidecar state file is updated with the outcome.
    """
    plan_path = Path(plan_path).expanduser()
    plan = validate_plan_file(plan_path, repo_path=repo_path, check_files=False)
    step = _find_step(plan, step_id)
    if step is None:
        raise KeyError(f"step {step_id!r} not in plan {plan['plan_id']!r}")

    root = Path(repo_path).expanduser() if repo_path else Path.cwd()

    # --- Dependency gate ---
    state = load_state(plan_path)
    unmet = deps_satisfied(state, step.get("depends_on", []) or [])
    if unmet:
        result = StepResult(
            step_id=step_id,
            status="blocked",
            tier=_pick_tier(step),
            attempted_tiers=[],
            files_changed=[],
            model_notes="",
            acceptance_passed=False,
            acceptance_output=f"blocked on: {', '.join(unmet)}",
            bandit_reward=0.0,
            dry_run=dry_run,
        )
        return result

    # --- Guard check (pre-dispatch, file-path based) ---
    active_guards = guards if guards is not None else load_guards()
    pre_matches = check_pre_dispatch(step["files"], root, active_guards)
    pre_escalation = first_escalation(pre_matches)
    if pre_escalation:
        mark_step(
            plan_path, step_id, "escalated",
            tier=_pick_tier(step),
            acceptance_output=pre_escalation.as_message(),
            notes="guard triggered before dispatch",
        )
        return StepResult(
            step_id=step_id,
            status="escalated",
            tier=_pick_tier(step),
            attempted_tiers=[],
            files_changed=[],
            model_notes=pre_escalation.as_message(),
            acceptance_passed=False,
            acceptance_output=pre_escalation.as_message(),
            bandit_reward=0.0,
            dry_run=dry_run,
        )

    # Don't re-run a done step unless caller explicitly reruns with a different id.
    if step_status(state, step_id) == "done":
        result = StepResult(
            step_id=step_id,
            status="done",
            tier=_pick_tier(step),
            attempted_tiers=[],
            files_changed=[],
            model_notes="already complete",
            acceptance_passed=True,
            acceptance_output="(skipped — already done)",
            bandit_reward=0.0,
            dry_run=dry_run,
        )
        return result

    # --- Resolve executor ---
    if chat_fn is None:
        from ..llm.litellm_adapter import get_provider
        chat_fn = get_provider().chat

    if reward_fn is None:
        from ..router import bandit as _bandit
        from ..store import SkillStore
        _store = SkillStore()
        def reward_fn(tier: str, task_class: str, domain: str, success: float) -> None:
            _bandit.record_reward(_store, tier, task_class, domain, success)

    # --- Build prompt ---
    bundle = build_bundle(
        root,
        files=step["files"],
        protocols_ref=step.get("protocols_ref"),
        pattern_ref=step.get("pattern_ref"),
    )
    user_prompt = _build_user_prompt(plan, step, bundle)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    primary_tier = _pick_tier(step)
    # force_tier_smart guard upgrades mid-tier steps to tier_smart.
    if any(m.rule.action == "force_tier_smart" for m in pre_matches):
        primary_tier = "tier_smart"
    attempted: list[str] = []
    last_output = ""
    changed: list[str] = []
    notes = ""
    acceptance_passed = False
    post_escalation: str | None = None

    mark_step(plan_path, step_id, "running", tier=primary_tier)

    # Retry policy: primary tier → if acceptance fails, escalate once to tier_smart.
    tiers_to_try = [primary_tier]
    if primary_tier != "tier_smart":
        tiers_to_try.append("tier_smart")

    for tier in tiers_to_try:
        attempted.append(tier)
        try:
            raw = chat_fn(messages, tier=tier, max_tokens=4096, temperature=0.2)
        except Exception as e:  # noqa: BLE001
            last_output = f"LLM call failed on {tier}: {e}"
            continue
        try:
            payload = _parse_model_output(raw)
            changes = payload.get("files") or {}
            notes = payload.get("notes", "")
        except (json.JSONDecodeError, ValueError) as e:
            last_output = f"could not parse model output on {tier}: {e}\nraw: {raw[:500]}"
            continue

        # Post-dispatch content guards (e.g. AI attribution strings).
        post_matches = check_post_dispatch(changes, root, active_guards)
        post_esc = first_escalation(post_matches)
        if post_esc:
            post_escalation = post_esc.as_message()
            last_output = post_escalation
            # Do not write, do not retry — a content guard means "human decides".
            break

        declared = set(step["files"])
        if dry_run:
            changed = sorted(changes.keys() & declared)
            acceptance_passed = True
            last_output = f"(dry_run — would change {len(changed)} files)"
            break

        try:
            changed = _apply_changes(root, declared, changes)
        except ValueError as e:
            last_output = f"scope violation on {tier}: {e}"
            continue

        acceptance_passed, last_output = _run_acceptance(step["acceptance"], root)
        if acceptance_passed:
            break
        # Else: loop to next tier (escalate) without re-applying changes — the
        # next tier will produce its own files.

    # --- Record outcome ---
    if post_escalation:
        final_status = "escalated"
        reward = 0.0
    elif acceptance_passed:
        final_status = "done"
        reward = 1.0
    else:
        final_status = "failed"
        reward = 0.0

    final_tier = attempted[-1] if attempted else primary_tier
    try:
        reward_fn(final_tier, step["kind"], plan["plan_id"], reward)
    except Exception:  # noqa: BLE001
        # Reward recording must never break the executor.
        pass

    mark_step(
        plan_path,
        step_id,
        final_status,
        tier=final_tier,
        acceptance_output=last_output,
        notes=notes,
    )

    return StepResult(
        step_id=step_id,
        status=final_status,
        tier=final_tier,
        attempted_tiers=attempted,
        files_changed=changed,
        model_notes=notes,
        acceptance_passed=acceptance_passed,
        acceptance_output=last_output,
        bandit_reward=reward,
        dry_run=dry_run,
    )
