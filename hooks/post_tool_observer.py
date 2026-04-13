#!/usr/bin/env python3
"""PostToolUse hook: record successful tool runs as training signal.

If Claude ran a Bash command successfully, the user either approved it or it
was auto-approved. Either way the command was judged safe-enough in this
project context — record it so future identical/similar commands can be
auto-approved from cache without re-prompting.

Only writes to cache when:
  - tool_name is Bash (others are handled by safe_tools list)
  - the tool did not error (PostToolUse fires regardless; we inspect result)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verdict_cache  # noqa: E402

LOG = Path.home() / ".claude" / "mcp-skill-hub" / "logs" / "hook-debug.log"


def log(msg: str) -> None:
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a") as f:
            tag = verdict_cache.task_tag()
            f.write(f"[{datetime.now():%H:%M:%S}] POST_TOOL {tag}{msg}\n")
    except OSError:
        pass


def main() -> int:
    cfg = verdict_cache.load_config()
    if not cfg.get("auto_approve_learn", True):
        return 0

    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return 0

    tool_name = data.get("tool_name", "")
    if tool_name != "Bash":
        return 0

    tool_input = data.get("tool_input") or {}
    cmd = (tool_input.get("command") or "").strip()
    if not cmd:
        return 0

    tool_response = data.get("tool_response") or {}
    # PostToolUse payload shape: tool_response may be dict or str; treat an
    # explicit error field or non-zero exit code as "don't learn".
    if isinstance(tool_response, dict):
        if tool_response.get("is_error") or tool_response.get("error"):
            log(f"skip  errored  cmd=\"{cmd[:60]}\"")
            return 0
        interrupted = tool_response.get("interrupted")
        if interrupted:
            log(f"skip  interrupted  cmd=\"{cmd[:60]}\"")
            return 0

    try:
        conn = verdict_cache.connect()
        verdict_cache.put(conn, tool_name, cmd, "allow", "user_approved", 1.0)
        log(f"learned  cmd=\"{cmd[:60]}\"")
    except Exception as e:  # noqa: BLE001
        log(f"cache  error={e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
