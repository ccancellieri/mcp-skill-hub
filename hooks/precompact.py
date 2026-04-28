#!/usr/bin/env python3
"""PreCompact hook — snapshot routing/tool-chain state before /compact wipes it.

Compaction summarises the conversation and discards intermediate state, which
includes the in-flight routing-bandit reward signal and the recent tool-chain
window used for habit mining. We persist a snapshot keyed on session_id so
post-compact code can pick the trail back up.
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
            f.write(f"[{datetime.now():%H:%M:%S}] PRECOMPACT {msg}\n")
    except OSError:
        pass


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return 0

    session_id = data.get("session_id", "")
    transcript = data.get("transcript_path", "")
    trigger = data.get("trigger", "")  # "manual" or "auto" per Claude Code docs

    if not session_id:
        _debug("skip  reason=no_session_id")
        return 0

    if not CLI.exists():
        _debug(f"skip  cli_missing={CLI}")
        return 0

    cmd = [
        str(CLI), "precompact_snapshot",
        "--session-id", session_id,
        "--trigger", trigger or "unknown",
    ]
    if transcript:
        cmd.extend(["--transcript", transcript])

    t0 = time.monotonic()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        _debug("error  cli timed out")
        return 0
    except OSError as exc:
        _debug(f"error  {exc}")
        return 0

    ms = int((time.monotonic() - t0) * 1000)
    _debug(f"done  exit={result.returncode}  time={ms}ms  trigger={trigger}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
