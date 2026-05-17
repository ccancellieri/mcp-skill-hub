"""Render the fan-out result as a directive the active Claude can paste back.

The directive tells the active Claude to emit ONE message containing N
`Agent({ ... })` tool calls — that's how Claude Code dispatches subagents
in parallel.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class DispatchSpec:
    """One Agent({...}) call to be emitted by the active Claude."""
    description: str
    prompt: str
    worktree_path: str
    branch: str
    task_id: int


def _shape_call(spec: DispatchSpec) -> str:
    return (
        "Agent({\n"
        f"  description: {json.dumps(spec.description)},\n"
        f"  prompt: {json.dumps(spec.prompt)},\n"
        '  subagent_type: "general-purpose",\n'
        '  isolation: "worktree",\n'
        "})"
    )


def render_directive(
    specs: list[DispatchSpec],
    group_id: str,
    *,
    rollup_tool: str = "fanout_status",
    close_tool: str = "fanout_close",
) -> str:
    """Plain-text directive: dispatch all agents in ONE message, then rollup.

    The directive intentionally uses *zero* AI-tooling paths in its prose
    (no `~/.claude/`, no plan filenames, no session ids) — the only paths
    referenced are the per-worktree paths the user already owns.
    """
    if not specs:
        return f"Fanout group {group_id}: no tasks created. Nothing to dispatch."

    calls = "\n\n".join(_shape_call(s) for s in specs)
    n = len(specs)
    summary_rows = "\n".join(
        f"  - task #{s.task_id} → {s.worktree_path} (branch {s.branch})"
        for s in specs
    )
    return (
        f"Fanout group `{group_id}` is prepped: {n} worktrees ready, {n} skill-hub tasks open.\n\n"
        f"{summary_rows}\n\n"
        f"Now dispatch all {n} agents IN A SINGLE MESSAGE so they run concurrently. "
        f"Paste the following block as one message containing {n} Agent tool calls:\n\n"
        f"```\n{calls}\n```\n\n"
        f"After all agents return:\n"
        f"  - call `{rollup_tool}(group_id={json.dumps(group_id)})` for a progress roll-up\n"
        f"  - call `{close_tool}(group_id={json.dumps(group_id)})` to close the group when merged\n"
    )


__all__ = ["DispatchSpec", "render_directive"]
