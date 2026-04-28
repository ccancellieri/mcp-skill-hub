#!/usr/bin/env python3
"""SubagentStart / SubagentStop hook — log subagent lifecycle into session_log.

Both events carry ``agent_id`` and ``agent_type``; SubagentStop also carries
``agent_transcript_path`` and the agent's final response text. Recording them
gives ``session_stats()`` a way to break tool usage out by agent_type so the
main thread vs Explore/Plan/etc can be analysed separately.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

DEBUG_LOG = Path.home() / ".claude" / "mcp-skill-hub" / "logs" / "hook-debug.log"


def _debug(msg: str) -> None:
    try:
        DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(DEBUG_LOG, "a") as f:
            f.write(f"[{datetime.now():%H:%M:%S}] SUBAGENT   {msg}\n")
    except OSError:
        pass


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return 0

    event = data.get("hook_event_name", "")
    session_id = data.get("session_id", "")
    agent_id = data.get("agent_id", "")
    agent_type = data.get("agent_type", "")

    if not session_id or not agent_id:
        _debug(f"skip  event={event}  reason=missing_ids")
        return 0

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from skill_hub.store import SkillStore
    except ImportError as exc:
        _debug(f"skip  import_error={exc}")
        return 0

    try:
        store = SkillStore()
        try:
            store.log_session_subagent(
                session_id=session_id,
                agent_id=agent_id,
                agent_type=agent_type,
                event=event,
                transcript_path=data.get("agent_transcript_path", ""),
            )
        finally:
            store.close()
    except Exception as exc:  # noqa: BLE001
        _debug(f"db_error  {exc}")
        return 0

    _debug(f"logged  event={event}  type={agent_type}  id={agent_id[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
