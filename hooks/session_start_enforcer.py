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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verdict_cache  # noqa: E402

DEBUG_LOG = Path.home() / ".claude" / "mcp-skill-hub" / "logs" / "hook-debug.log"
RESUME_MARKER = Path.home() / ".claude" / "mcp-skill-hub" / "state" / "needs_resume.json"


def consume_resume_marker() -> str:
    """If a previous session ended on api_error, return a resume reminder and
    delete the marker. Empty string otherwise."""
    if not RESUME_MARKER.exists():
        return ""
    try:
        data = json.loads(RESUME_MARKER.read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    try:
        RESUME_MARKER.unlink()
    except OSError:
        pass
    prev = data.get("session_id", "?")
    plan = data.get("plan", "")
    at = data.get("at", "")
    msg = (
        f"RESUME: the previous session ({prev}) ended on a transient API error at {at}. "
    )
    if plan:
        msg += f"Active plan: ~/.claude/plans/{plan}. Continue from where it left off; re-read the plan file first."
    else:
        msg += "Inspect the last transcript and resume the interrupted work."
    return msg


def _check_profile_drift() -> str:
    """Compare active profile (if any) to live enabledPlugins. Return advisory
    string on mismatch, empty otherwise. Best-effort — never raises."""
    try:
        # Lazy import — hooks must stay fast even if skill-hub isn't installed.
        from skill_hub.store import SkillStore
        from skill_hub import profiles as _prof
    except Exception:
        return ""
    try:
        drift = _prof.detect_profile_drift(SkillStore())
    except Exception:
        return ""
    if not drift:
        return ""
    missing = len(drift.get("missing") or {})
    unexpected = len(drift.get("unexpected") or {})
    return (
        f"PROFILE DRIFT: active profile {drift['profile']!r} does not match "
        f"~/.claude/settings.json enabledPlugins "
        f"({missing} missing, {unexpected} unexpected). "
        f"Run `switch_profile(name={drift['profile']!r})` then restart."
    )


def log(msg: str):
    try:
        DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%H:%M:%S")
        with open(DEBUG_LOG, "a") as f:
            tag = verdict_cache.task_tag()
            f.write(f"[{ts}] [  0.0s] ENFORCER   {tag}{msg}\n")
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

    resume_msg = consume_resume_marker()
    if resume_msg:
        log(f"RESUME marker consumed  msg=\"{resume_msg[:80]}\"")

    drift_msg = _check_profile_drift()
    if drift_msg:
        log(f"PROFILE drift detected  msg=\"{drift_msg[:100]}\"")

    log(f"injecting session-start reminder  log_cmd=\"{log_cmd}\"")

    system_msg = (
        "SESSION START:\n"
        "1. Read project .memory/index.md if it exists in the working directory\n"
        "2. Follow CLAUDE.md multi-level context protocol\n"
        "\n"
        f"Hook activity log: {log_cmd}\n"
        "Mention the log command to the user so they can follow local LLM activity."
    )
    if resume_msg:
        system_msg = resume_msg + "\n\n" + system_msg
    if drift_msg:
        system_msg = drift_msg + "\n\n" + system_msg

    print(json.dumps({"decision": "allow", "systemMessage": system_msg}))


if __name__ == "__main__":
    main()
