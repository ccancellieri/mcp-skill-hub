#!/usr/bin/env python3
"""SessionEnd hook — once-per-session work that Stop can't safely do.

Stop fires after every model turn; SessionEnd (Claude Code 1.0.85) fires
once when the session truly ends. We use it for the things that should
*not* run per-turn:
  - persist the final L1 summary into ``session:log``
  - dispatch the ``on_session_end`` plugin hook
  - rotate the in-process session id

The per-turn work (Context Bridge tool capture, conditional memory save)
stays on the existing Stop hook (`session_end.sh` → `session_end.py`).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
IS_WINDOWS = sys.platform == "win32"
CLI = SCRIPT_DIR / ".venv" / ("Scripts" if IS_WINDOWS else "bin") / (
    "skill-hub-cli.exe" if IS_WINDOWS else "skill-hub-cli"
)
DEBUG_LOG = Path.home() / ".claude" / "mcp-skill-hub" / "logs" / "hook-debug.log"


def _debug(msg: str) -> None:
    try:
        DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(DEBUG_LOG, "a") as f:
            f.write(f"[{datetime.now():%H:%M:%S}] SESS_END   {msg}\n")
    except OSError:
        pass


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return 0

    session_id = data.get("session_id", "")
    reason = data.get("reason", "")  # "exit", "clear", "logout", etc.
    last_msg = (data.get("last_assistant_message", "") or "")[:1000]

    if not session_id:
        _debug("skip  reason=no_session_id")
        return 0

    if not CLI.exists():
        _debug(f"skip  cli_missing={CLI}")
        return 0

    cmd = [
        str(CLI), "session_close",
        "--session-id", session_id,
        "--reason", reason or "unknown",
    ]
    if last_msg:
        cmd.extend(["--summary", last_msg])

    t0 = time.monotonic()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        _debug("error  cli timed out")
        return 0
    except OSError as exc:
        _debug(f"error  {exc}")
        return 0

    ms = int((time.monotonic() - t0) * 1000)
    _debug(f"done  exit={result.returncode}  time={ms}ms  reason={reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
