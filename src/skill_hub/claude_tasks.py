"""Pure helpers for observing Claude Code task tools.

No DB access, no imports of skill_hub.store or skill_hub.server.
Safe to import in any context — especially the hook path where
startup latency must stay minimal.
"""
from __future__ import annotations

import hashlib

CLAUDE_TASK_TOOLS = frozenset(
    {"TodoWrite", "TaskCreate", "TaskUpdate", "TaskComplete", "TaskStop"}
)

_COMPLETION_STATUSES = frozenset(
    {"completed", "done", "complete", "cancelled", "canceled", "stopped"}
)


def _first_nonempty(*values: object) -> str:
    """Return the first non-empty string value, or ''."""
    for v in values:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def parse_claude_tasks(
    tool_name: str,
    tool_input: dict,
    tool_response: dict | None = None,
) -> list[dict]:
    """Parse Claude tool I/O into a list of task dicts.

    Each dict has keys: ``identity``, ``title``, ``status``, ``claude_id``.
    Never raises; returns [] for unknown shapes or errors.
    """
    try:
        return _parse(tool_name, tool_input, tool_response or {})
    except Exception:  # noqa: BLE001
        return []


def _parse(
    tool_name: str,
    tool_input: dict,
    tool_response: dict,
) -> list[dict]:
    if tool_name == "TodoWrite":
        todos = tool_input.get("todos")
        if not isinstance(todos, list):
            return []
        results = []
        for item in todos:
            if not isinstance(item, dict):
                continue
            title = _first_nonempty(
                item.get("content"),
                item.get("activeForm"),
                item.get("description"),
            )
            status = item.get("status") or "pending"
            results.append(
                {"identity": title, "title": title, "status": status, "claude_id": None}
            )
        return results

    if tool_name in ("TaskCreate", "TaskUpdate"):
        title = _first_nonempty(
            tool_input.get("description"),
            tool_input.get("title"),
            tool_input.get("prompt"),
            tool_input.get("subagent_type"),
        )
        claude_id = (
            tool_input.get("task_id")
            or tool_response.get("task_id")
            or tool_response.get("id")
        )
        if claude_id is not None:
            claude_id = str(claude_id)
        status = tool_input.get("status") or "open"
        return [{"identity": title, "title": title, "status": status, "claude_id": claude_id}]

    if tool_name == "TaskComplete":
        claude_id = tool_input.get("task_id") or tool_response.get("task_id")
        if claude_id is not None:
            claude_id = str(claude_id)
        title = _first_nonempty(
            tool_input.get("description"),
            tool_input.get("title"),
        )
        return [{"identity": title, "title": title, "status": "completed", "claude_id": claude_id}]

    if tool_name == "TaskStop":
        claude_id = tool_input.get("task_id") or tool_response.get("task_id")
        if claude_id is not None:
            claude_id = str(claude_id)
        title = _first_nonempty(
            tool_input.get("description"),
            tool_input.get("title"),
        )
        return [{"identity": title, "title": title, "status": "stopped", "claude_id": claude_id}]

    return []


def stable_key(
    identity: str,
    cwd: str = "",
    branch: str = "",
    claude_id: str | None = None,
) -> str:
    """Compute a stable dedup key for a Claude task.

    If ``claude_id`` is provided, returns ``cid:<claude_id>`` (globally unique).
    Otherwise returns a sha256 hex[:16] of ``identity|cwd|branch`` prefixed
    with ``txt:``.
    """
    if claude_id:
        return f"cid:{claude_id}"
    raw = f"{identity.strip().lower()}|{cwd}|{branch}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"txt:{digest}"
