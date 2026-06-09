"""Render the fan-out result as a directive the active Claude can paste back.

The directive tells the active Claude to emit ONE message containing N
`Agent({ ... })` tool calls — that's how Claude Code dispatches subagents
in parallel.

Each worktree is dispatched to a *specialized* implementer agent
(``team-code-implementer``, the sonnet build role) rather than a generic
``general-purpose`` agent — the orchestration layer converged on the native
``/team`` roles, so fan-out is the issue→worktree front-end that feeds them.
For architecture-heavy issues an operator can instead drive the full
``/team implement`` pipeline (design → build → verify → clean → PR) inside a
single worktree.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

# The specialized build role from the /team orchestration layer (sonnet).
# Replaces the historical hardcoded "general-purpose" subagent so fan-out work
# inherits the right-model-for-the-task routing instead of defaulting to opus.
DEFAULT_SUBAGENT_TYPE = "team-code-implementer"


@dataclass
class DispatchSpec:
    """One Agent({...}) call to be emitted by the active Claude."""
    description: str
    prompt: str
    worktree_path: str
    branch: str
    task_id: int


def _shape_call(spec: DispatchSpec, subagent_type: str) -> str:
    return (
        "Agent({\n"
        f"  description: {json.dumps(spec.description)},\n"
        f"  prompt: {json.dumps(spec.prompt)},\n"
        f"  subagent_type: {json.dumps(subagent_type)},\n"
        '  isolation: "worktree",\n'
        "})"
    )


def render_directive(
    specs: list[DispatchSpec],
    group_id: str,
    *,
    rollup_tool: str = "fanout_status",
    close_tool: str = "fanout_close",
    subagent_type: str = DEFAULT_SUBAGENT_TYPE,
) -> str:
    """Plain-text directive: dispatch all agents in ONE message, then rollup.

    The directive intentionally uses *zero* AI-tooling paths in its prose
    (no `~/.claude/`, no plan filenames, no session ids) — the only paths
    referenced are the per-worktree paths the user already owns.

    ``subagent_type`` selects the specialized agent each worktree is handed to;
    it defaults to the sonnet build role so fan-out work is routed by the same
    model·effort policy as the rest of the /team layer.
    """
    if not specs:
        return f"Fanout group {group_id}: no tasks created. Nothing to dispatch."

    calls = "\n\n".join(_shape_call(s, subagent_type) for s in specs)
    n = len(specs)
    summary_rows = "\n".join(
        f"  - task #{s.task_id} → {s.worktree_path} (branch {s.branch})"
        for s in specs
    )
    return (
        f"Fanout group `{group_id}` is prepped: {n} worktrees ready, {n} skill-hub tasks open.\n\n"
        f"{summary_rows}\n\n"
        f"Now dispatch all {n} agents IN A SINGLE MESSAGE so they run concurrently. "
        f"Each runs as the specialized `{subagent_type}` role (right-model routing). "
        f"Paste the following block as one message containing {n} Agent tool calls:\n\n"
        f"```\n{calls}\n```\n\n"
        f"For an architecture-heavy issue, run `/team implement <issue>` inside its "
        f"worktree instead, to get the full design → build → verify → clean → PR pipeline.\n\n"
        f"After all agents return:\n"
        f"  - call `{rollup_tool}(group_id={json.dumps(group_id)})` for a progress roll-up\n"
        f"  - call `{close_tool}(group_id={json.dumps(group_id)})` to close the group when merged\n"
    )


__all__ = ["DEFAULT_SUBAGENT_TYPE", "DispatchSpec", "render_directive"]
