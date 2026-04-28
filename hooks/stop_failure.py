#!/usr/bin/env python3
"""StopFailure hook — append API-error events to api-errors.jsonl.

Fires when a turn ends because of an API/network/rate-limit error rather
than the model finishing normally. Captures the failure for the dashboard's
monitoring tab so we can spot patterns (e.g. recurring 529 overload bursts)
without scraping debug logs by hand.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

LOG_DIR = Path.home() / ".claude" / "mcp-skill-hub" / "logs"
ERRORS_FILE = LOG_DIR / "api-errors.jsonl"
DEBUG_LOG = LOG_DIR / "hook-debug.log"


def _debug(msg: str) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(DEBUG_LOG, "a") as f:
            f.write(f"[{datetime.now():%H:%M:%S}] STOP_FAIL  {msg}\n")
    except OSError:
        pass


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return 0

    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "session_id": data.get("session_id", ""),
        "transcript_path": data.get("transcript_path", ""),
        "agent_id": data.get("agent_id", ""),
        "agent_type": data.get("agent_type", ""),
        "stop_hook_active": data.get("stop_hook_active", False),
        "error": data.get("error") or data.get("error_message") or "",
        "error_type": data.get("error_type", ""),
    }

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(ERRORS_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as exc:
        _debug(f"write_error  {exc}")
        return 0

    _debug(f"recorded  session={record['session_id'][:12]}  err={(record['error'] or '')[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
