#!/usr/bin/env python3
"""PostCompact hook — run optimize_memory after compaction trims the context.

Compaction is the natural moment to prune: prior exchanges have just been
summarised so duplicate L0/L1 vectors are stalest now. Runs optimize_memory
in dry-run mode by default so the user sees recommendations as a system
message; flip cfg `postcompact_optimize_apply` to True to actually mutate.
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
            f.write(f"[{datetime.now():%H:%M:%S}] POSTCOMPACT {msg}\n")
    except OSError:
        pass


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return 0

    session_id = data.get("session_id", "")

    if not CLI.exists():
        _debug(f"skip  cli_missing={CLI}")
        return 0

    cmd = [str(CLI), "postcompact_optimize", "--session-id", session_id]

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
    _debug(f"done  exit={result.returncode}  time={ms}ms")

    if result.returncode != 0 or not result.stdout.strip():
        return 0

    # If the CLI returned a systemMessage payload, surface it to Claude so
    # the user sees the recommendations in their stream.
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return 0

    msg = payload.get("systemMessage")
    if msg:
        out = {"systemMessage": msg}
        hso = payload.get("hookSpecificOutput")
        if hso:
            out["hookSpecificOutput"] = hso
        print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
