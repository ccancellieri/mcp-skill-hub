"""Session → task auto-bind: resume an open task on match, else create a new one.

Called from the UserPromptSubmit hook (`session_start_enforcer.py`) on the
first prompt of each Claude Code session. Always refreshes `active_task.json`
so auto_approve never attributes a new session's tool calls to a stale task.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ACTIVE_TASK_MARKER = (
    Path.home() / ".claude" / "mcp-skill-hub" / "state" / "active_task.json"
)


def _get_config() -> dict:
    """Read the 4 session-bind config keys. Safe defaults if skill_hub unimportable."""
    try:
        from skill_hub import config as _cfg
        return {
            "enabled": bool(_cfg.get("session_task_auto_create_enabled")),
            "strategy": str(_cfg.get("session_task_match_strategy") or "hybrid"),
            "window_days": int(_cfg.get("session_task_match_window_days") or 7),
            "semantic_threshold": float(_cfg.get("session_task_semantic_threshold") or 0.75),
        }
    except Exception:
        return {"enabled": True, "strategy": "hybrid",
                "window_days": 7, "semantic_threshold": 0.75}


def _embed_message(message: str) -> list[float]:
    """Best-effort embedding — returns [] on any failure."""
    try:
        from skill_hub.embeddings import embed
        return embed(message) or []
    except Exception:
        return []


def _derive_title(message: str) -> str:
    """First sentence or 120 chars, whichever is shorter."""
    first_line = (message.strip().splitlines() or [""])[0][:200]
    m = re.search(r"[.!?]", first_line)
    title = first_line[:m.start()] if m and m.start() > 20 else first_line
    return title.strip()[:120] or "Conversation"


def _write_marker(task_id: int, session_id: str, title: str) -> None:
    try:
        ACTIVE_TASK_MARKER.parent.mkdir(parents=True, exist_ok=True)
        ACTIVE_TASK_MARKER.write_text(json.dumps({
            "task_id": task_id,
            "session_id": session_id,
            "title": title,
            "auto_approve": None,
            "options": {},
        }, indent=2))
    except OSError:
        pass


def bind_session_to_task(
    session_id: str,
    message: str,
    cwd: str,
    branch: str,
    store,
) -> tuple[str, int, str, str | None]:
    """Resume-or-create. Returns (action, task_id, title, match_reason).

    action ∈ {"resumed", "created", "skipped"}.
    match_reason is None for "created"/"skipped", else "cwd+branch" or "semantic:<score>".
    """
    cfg = _get_config()
    if not cfg["enabled"] or cfg["strategy"] == "off":
        return "skipped", 0, "", None
    if not session_id:
        return "skipped", 0, "", None

    strategy = cfg["strategy"]
    window = cfg["window_days"]

    if strategy in ("hybrid", "cwd_branch") and cwd:
        row = store.find_resumable_task_by_cwd_branch(cwd, branch or "", window)
        if row is not None:
            tid = int(row["id"])
            title = row["title"] or "(untitled)"
            store.bind_task_to_session(tid, session_id)
            _write_marker(tid, session_id, title)
            return "resumed", tid, title, "cwd+branch"

    if strategy in ("hybrid", "semantic") and len(message.strip()) >= 30:
        vec = _embed_message(message)
        if vec:
            hit = store.find_resumable_task_semantic(vec, window, cfg["semantic_threshold"])
            if hit is not None:
                row, score = hit
                tid = int(row["id"])
                title = row["title"] or "(untitled)"
                store.bind_task_to_session(tid, session_id)
                _write_marker(tid, session_id, title)
                return "resumed", tid, title, f"semantic:{score:.2f}"

    title = _derive_title(message)
    vec = _embed_message(message) if message.strip() else []
    tid = store.save_task(
        title=title, summary=message[:500], vector=vec,
        context="", tags="", session_id=session_id,
        cwd=cwd or "", branch=branch or "",
    )
    _write_marker(tid, session_id, title)
    return "created", tid, title, None
