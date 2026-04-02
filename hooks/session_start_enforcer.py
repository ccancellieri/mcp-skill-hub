#!/usr/bin/env python3
"""Session-start protocol enforcer hook (cross-platform).

On the first user prompt of each session, injects a systemMessage
reminding Claude to execute the mandatory session-start checklist.

Cost: 0 LLM tokens. Just a file-existence check (~1ms after first call).
"""

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

DEBUG_LOG = Path.home() / ".claude" / "mcp-skill-hub" / "logs" / "hook-debug.log"


def log(msg: str):
    try:
        DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%H:%M:%S")
        with open(DEBUG_LOG, "a") as f:
            f.write(f"[{ts}] [  0.0s] ENFORCER   {msg}\n")
    except OSError:
        pass


def main():
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return

    session_id = data.get("session_id", "unknown")
    flag = os.path.join(tempfile.gettempdir(), f"claude-session-started-{session_id}")

    if os.path.exists(flag):
        # Not the first message — exit silently
        return

    # First message in session — inject checklist
    log(f"NEW SESSION  id={session_id}")

    # Mark session as started
    with open(flag, "w") as f:
        f.write("")

    # Clean up flags older than 24h (best-effort)
    try:
        import time
        now = time.time()
        tmp = tempfile.gettempdir()
        pruned = 0
        for name in os.listdir(tmp):
            if name.startswith("claude-session-started-"):
                path = os.path.join(tmp, name)
                if now - os.path.getmtime(path) > 86400:
                    os.remove(path)
                    pruned += 1
        if pruned:
            log(f"pruned {pruned} stale session flags")
    except OSError:
        pass

    log_path = str(DEBUG_LOG)
    # Detect platform for log command
    if sys.platform == "win32":
        log_cmd = f"powershell: Get-Content -Wait {log_path}"
    else:
        log_cmd = f"tail -f {log_path}"

    log(f"injecting session-start reminder  log_cmd=\"{log_cmd}\"")

    print(
        json.dumps(
            {
                "decision": "allow",
                "systemMessage": (
                    "SESSION START:\n"
                    "1. Read project .memory/index.md if it exists in the working directory\n"
                    "2. Follow CLAUDE.md multi-level context protocol\n"
                    "\n"
                    f"Hook activity log: {log_cmd}\n"
                    "Mention the log command to the user so they can follow local LLM activity."
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
