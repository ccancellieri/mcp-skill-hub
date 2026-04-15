"""Transparent Opus-authoring runners for the plan-aware executor.

The hub tries runners in priority order and uses the first that works:

  1. in_session   — we're inside Claude Code; return a directive so the
                    calling agent (Opus) authors the plan itself. Zero cost.
  2. cli          — spawn `claude -p ...` with subscription OAuth. Max plan.
  3. sdk          — import claude_agent_sdk and run programmatically. Max plan.
  4. api          — fall back to litellm with tier_planner (API tokens).

The user overrides with HUB_PLAN_RUNNER=in_session|cli|sdk|api. Otherwise
resolution is silent — they see only "authored via <runner>" in the footer.

Each runner returns a ``RunnerResult`` with either the YAML text produced or
a ``directive`` string that the caller must display to the user/agent.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal


RunnerName = Literal["in_session", "cli", "sdk", "api"]
RUNNER_ORDER: tuple[RunnerName, ...] = ("in_session", "cli", "sdk", "api")


@dataclass
class RunnerResult:
    runner: RunnerName
    yaml_text: str | None       # None when a directive is returned instead
    directive: str | None       # set for in_session; instructs the caller
    raw_output: str             # verbatim runner stdout, for debugging


class RunnerUnavailable(RuntimeError):
    """Raised when the runner isn't available on this system."""


class RunnerFailed(RuntimeError):
    """Raised when the runner ran but produced no usable output."""


# ---- runner implementations -------------------------------------------------

def _build_system_prompt(schema_hint: str) -> str:
    return (
        "You are the PLANNER for a plan-aware executor. Given a goal and a "
        "target repository, emit ONLY a YAML document conforming to the "
        "plan_executor schema. No prose, no markdown fence — just the YAML "
        "document starting with 'plan_id:'.\n\n"
        f"Schema:\n{schema_hint}\n\n"
        "Rules:\n"
        "- Every step must be atomic: one concern, <=5 files.\n"
        "- Prefer small steps (kind=boilerplate|tests|docs) over large ones.\n"
        "- Use depends_on to order steps; parallel-safe steps may omit it.\n"
        "- Set protocols_ref for kind=architecture|integration.\n"
        "- acceptance should be a runnable command (pytest/pyright/ruff) "
        "where possible; otherwise a short heuristic.\n"
    )


def _build_user_prompt(goal: str, repo_path: Path) -> str:
    return (
        f"GOAL: {goal}\n"
        f"REPO: {repo_path}\n\n"
        "Explore the repo as needed, then emit the plan YAML."
    )


def run_in_session(
    goal: str,
    repo_path: Path,
    schema_hint: str,
    plan_out_path: Path,
) -> RunnerResult:
    """Return a directive for the *current* Claude Code session to author
    the plan directly. Available only when CLAUDECODE=1 is set (we're being
    called from inside Claude Code)."""
    if os.environ.get("CLAUDECODE") != "1":
        raise RunnerUnavailable("not running inside Claude Code (CLAUDECODE!=1)")

    directive = (
        f"AUTHOR PLAN IN-SESSION:\n"
        f"  goal: {goal}\n"
        f"  repo_path: {repo_path}\n"
        f"  plan_out_path: {plan_out_path}\n\n"
        f"Instructions for the agent currently handling this conversation:\n"
        f"  1. Explore the repo using your Read/Glob/Grep tools.\n"
        f"  2. Write a YAML plan conforming to the schema below to {plan_out_path}.\n"
        f"  3. Then call the MCP tool `validate_plan(plan_path=\"{plan_out_path}\")`.\n"
        f"  4. If validation fails, fix and re-validate up to 2 times.\n\n"
        f"Schema:\n{schema_hint}"
    )
    return RunnerResult(
        runner="in_session",
        yaml_text=None,
        directive=directive,
        raw_output=directive,
    )


def run_cli(
    goal: str,
    repo_path: Path,
    schema_hint: str,
    plan_out_path: Path,
    *,
    timeout: float = 600.0,
) -> RunnerResult:
    """Spawn `claude -p` headlessly. Uses OAuth subscription when ANTHROPIC_API_KEY unset."""
    claude_bin = shutil.which("claude")
    if not claude_bin:
        raise RunnerUnavailable("`claude` CLI not on PATH")

    sysprompt = _build_system_prompt(schema_hint)
    user = _build_user_prompt(goal, repo_path)
    prompt = f"{sysprompt}\n\n{user}"

    try:
        proc = subprocess.run(
            [claude_bin, "-p", prompt, "--model", "opus"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise RunnerFailed(f"claude -p timed out after {timeout}s") from e
    except OSError as e:
        raise RunnerUnavailable(f"claude -p could not execute: {e}") from e

    if proc.returncode != 0:
        raise RunnerFailed(
            f"claude -p exited {proc.returncode}: {proc.stderr[-500:]}"
        )
    yaml_text = _strip_yaml_fences(proc.stdout)
    if not yaml_text.strip():
        raise RunnerFailed("claude -p returned empty output")
    return RunnerResult(
        runner="cli", yaml_text=yaml_text, directive=None, raw_output=proc.stdout
    )


def run_sdk(
    goal: str,
    repo_path: Path,
    schema_hint: str,
    plan_out_path: Path,
) -> RunnerResult:
    """Use the Claude Agent SDK programmatically. Same OAuth as the CLI."""
    try:
        import claude_agent_sdk  # type: ignore[import-not-found]
    except ImportError as e:
        raise RunnerUnavailable(f"claude_agent_sdk not installed: {e}") from e

    sysprompt = _build_system_prompt(schema_hint)
    user = _build_user_prompt(goal, repo_path)
    # SDK surface varies by version; we call the high-level query() if present.
    query = getattr(claude_agent_sdk, "query", None)
    if query is None:
        raise RunnerUnavailable("claude_agent_sdk has no query() entry point")

    try:
        raw = query(
            prompt=user,
            system=sysprompt,
            model="opus",
            cwd=str(repo_path),
        )
    except Exception as e:  # noqa: BLE001
        raise RunnerFailed(f"claude_agent_sdk call failed: {e}") from e

    text = raw if isinstance(raw, str) else str(raw)
    yaml_text = _strip_yaml_fences(text)
    if not yaml_text.strip():
        raise RunnerFailed("claude_agent_sdk returned empty output")
    return RunnerResult(
        runner="sdk", yaml_text=yaml_text, directive=None, raw_output=text
    )


def run_api(
    goal: str,
    repo_path: Path,
    schema_hint: str,
    plan_out_path: Path,
    *,
    chat_fn: Callable[..., str] | None = None,
) -> RunnerResult:
    """Fallback via litellm. Costs API tokens — emits a warning upstream.

    Uses tier_smart unless config maps tier_planner explicitly. Accepts an
    injected chat_fn for testing.
    """
    if chat_fn is None:
        # Double gate: config flag + env key. Default config has the flag off,
        # so Max-plan users never accidentally burn API tokens.
        from .. import config as _cfg
        if not _cfg.get("plan_api_runner_enabled"):
            raise RunnerUnavailable(
                "API runner disabled by config (plan_api_runner_enabled=False)"
            )
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RunnerUnavailable(
                "no ANTHROPIC_API_KEY set — API fallback disabled"
            )
        from ..llm.litellm_adapter import get_provider
        chat_fn = get_provider().chat

    sysprompt = _build_system_prompt(schema_hint)
    user = _build_user_prompt(goal, repo_path)
    messages = [
        {"role": "system", "content": sysprompt},
        {"role": "user", "content": user},
    ]
    # Try tier_planner first (config may map it to opus), fall back to tier_smart.
    for tier in ("tier_planner", "tier_smart"):
        try:
            raw = chat_fn(messages, tier=tier, max_tokens=8192, temperature=0.2)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
    else:
        raise RunnerFailed(f"litellm call failed: {last_err}")

    yaml_text = _strip_yaml_fences(raw)
    if not yaml_text.strip():
        raise RunnerFailed("litellm returned empty output")
    return RunnerResult(
        runner="api", yaml_text=yaml_text, directive=None, raw_output=raw
    )


# ---- orchestration ----------------------------------------------------------

_RUNNERS: dict[RunnerName, Callable[..., RunnerResult]] = {
    "in_session": run_in_session,
    "cli": run_cli,
    "sdk": run_sdk,
    "api": run_api,
}


def resolve_runner(
    goal: str,
    repo_path: Path,
    schema_hint: str,
    plan_out_path: Path,
    *,
    preferred: RunnerName | None = None,
    chat_fn: Callable[..., str] | None = None,
) -> RunnerResult:
    """Walk the runner chain, return the first that succeeds.

    If ``preferred`` is set (or HUB_PLAN_RUNNER env var is set), try only
    that runner and surface its errors directly. Otherwise try each runner
    in RUNNER_ORDER, silently skipping ``RunnerUnavailable`` ones.
    """
    override = preferred or os.environ.get("HUB_PLAN_RUNNER")
    if override:
        if override not in _RUNNERS:
            raise ValueError(
                f"invalid runner: {override!r} (choose from {RUNNER_ORDER})"
            )
        runner_fn = _RUNNERS[override]  # type: ignore[index]
        kwargs = {"chat_fn": chat_fn} if override == "api" else {}
        return runner_fn(goal, repo_path, schema_hint, plan_out_path, **kwargs)

    errors: list[str] = []
    for name in RUNNER_ORDER:
        runner_fn = _RUNNERS[name]
        kwargs = {"chat_fn": chat_fn} if name == "api" else {}
        try:
            return runner_fn(goal, repo_path, schema_hint, plan_out_path, **kwargs)
        except RunnerUnavailable as e:
            errors.append(f"{name}: unavailable ({e})")
            continue
        except RunnerFailed as e:
            errors.append(f"{name}: failed ({e})")
            continue
    raise RunnerFailed(
        "no runner succeeded:\n  " + "\n  ".join(errors)
    )


def _strip_yaml_fences(text: str) -> str:
    """Strip ```yaml fences and leading/trailing prose the model may add."""
    t = text.strip()
    if t.startswith("```"):
        # Drop the opening fence line.
        first_nl = t.find("\n")
        if first_nl >= 0:
            t = t[first_nl + 1 :]
    if t.endswith("```"):
        t = t[: t.rfind("```")].rstrip()
    return t
