#!/usr/bin/env python3
"""Stop hook: save session memory + stats when Claude finishes responding (cross-platform).

Smart routing:
  1. Local LLM generates memory from session context
  2. Quality check: if too much detail lost -> returns systemMessage
     asking Claude to write the memory instead
  3. Logs session stats (messages, interceptions, tokens saved)
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verdict_cache  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent.parent
IS_WINDOWS = sys.platform == "win32"
CLI = SCRIPT_DIR / ".venv" / ("Scripts" if IS_WINDOWS else "bin") / ("skill-hub-cli.exe" if IS_WINDOWS else "skill-hub-cli")
DEBUG_LOG = Path.home() / ".claude" / "mcp-skill-hub" / "logs" / "hook-debug.log"

_t0 = time.monotonic()
RESUME_MARKER = Path.home() / ".claude" / "mcp-skill-hub" / "state" / "needs_resume.json"


def check_api_error_and_mark(transcript: str, session_id: str) -> None:
    """If the transcript tail mentions an api_error, write a resume marker.

    session_start_enforcer reads it on the next launch and reminds Claude to
    continue the interrupted work.
    """
    if not transcript:
        return
    p = Path(transcript)
    if not p.exists():
        return
    try:
        # Read only the tail — transcripts can be large.
        with open(p, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 65536))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return

    markers = ("api_error", "Internal server error", "overloaded_error", "rate_limit")
    if not any(m in tail for m in markers):
        return

    # Try to capture the last assistant activity and the newest plan path.
    plans_dir = Path.home() / ".claude" / "plans"
    plan_hint = ""
    if plans_dir.exists():
        plans = sorted(plans_dir.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True)
        if plans:
            plan_hint = plans[0].name

    try:
        RESUME_MARKER.parent.mkdir(parents=True, exist_ok=True)
        RESUME_MARKER.write_text(json.dumps({
            "session_id": session_id,
            "transcript": str(p),
            "plan": plan_hint,
            "at": datetime.now().isoformat(timespec="seconds"),
        }, indent=2))
    except OSError:
        pass


def log(msg: str):
    try:
        DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        elapsed = time.monotonic() - _t0
        ts = datetime.now().strftime("%H:%M:%S")
        with open(DEBUG_LOG, "a") as f:
            tag = verdict_cache.task_tag()
            f.write(f"[{ts}] [{elapsed:6.1f}s] STOP       {tag}{msg}\n")
    except OSError:
        pass


def main():
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return

    session_id = data.get("session_id", "")
    last_msg = (data.get("last_assistant_message", "") or "")[:2000]
    last_msg = last_msg.replace("\n", " ").replace("\r", "").replace("\t", " ")
    transcript = data.get("transcript_path", "")
    stop_active = data.get("stop_hook_active", False)

    log(f"fired  session={session_id}  active={stop_active}  last_msg_len={len(last_msg)}  transcript={'yes' if transcript else 'no'}")

    # Mark for resume if this session ended on an API error (best-effort).
    try:
        check_api_error_and_mark(transcript, session_id)
    except Exception as e:  # noqa: BLE001
        log(f"resume_check_error  {e}")

    # Don't re-enter if already in a stop hook cycle
    if stop_active:
        log("skip  reason=already_active")
        return

    # Skip if no session context
    if not session_id and not last_msg:
        log("skip  reason=no_session_context")
        return

    # Run session_end via CLI
    cmd = [
        str(CLI), "session_end",
        "--session-id", session_id,
        "--last-message", last_msg,
        "--transcript", transcript,
    ]

    t_cli = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=45,
        )
    except subprocess.TimeoutExpired:
        log("error  session_end timed out after 45s")
        return
    except OSError as e:
        log(f"error  session_end failed: {e}")
        return

    cli_ms = int((time.monotonic() - t_cli) * 1000)
    log(f"cli  exit={result.returncode}  time={cli_ms}ms  stdout_len={len(result.stdout)}")

    if result.returncode != 0 or not result.stdout.strip():
        if result.stderr:
            log(f"cli_stderr  {result.stderr.strip()[:200]}")
        log(f"done  no_output  cli_time={cli_ms}ms")
        return

    try:
        cli_data = json.loads(result.stdout)
    except json.JSONDecodeError:
        log(f"error  invalid JSON: {result.stdout[:200]}")
        return

    has_system = bool(cli_data.get("systemMessage"))
    log(f"done  has_systemMessage={has_system}  cli_time={cli_ms}ms")

    # Forward systemMessage to Claude if present, but strip CLI's internal
    # `decision: "allow"` — Stop hook only accepts `decision: "block"` and
    # `(root): Invalid input` is raised otherwise.
    if has_system:
        out: dict = {"systemMessage": cli_data["systemMessage"]}
        hso = cli_data.get("hookSpecificOutput")
        if hso:
            out["hookSpecificOutput"] = hso
        if cli_data.get("decision") == "block" and cli_data.get("reason"):
            out["decision"] = "block"
            out["reason"] = cli_data["reason"]
        print(json.dumps(out))

    total_ms = int((time.monotonic() - _t0) * 1000)
    log(f"total_time={total_ms}ms")


if __name__ == "__main__":
    main()
