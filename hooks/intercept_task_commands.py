#!/usr/bin/env python3
"""UserPromptSubmit hook: intercept task/memory commands before Claude sees them (cross-platform).

Flow:
  1. Every user message passes through this hook BEFORE Claude sees it
  2. Python CLI does a fast embedding similarity check (~100ms):
     - Very long messages (>max_length) -> skip classification (coding questions)
     - Short messages -> embed and compare to canonical task phrases
     - Below similarity threshold -> allow through immediately
     - Above threshold -> call local LLM for precise classification (~2-5s)
  3. If it's a task command -> execute locally, block Claude (0 tokens used)
  4. If not -> dynamic context evaluation + prompt optimization via local LLM
"""

import json
import os
import re as _re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
IS_WINDOWS = sys.platform == "win32"
CLI = SCRIPT_DIR / ".venv" / ("Scripts" if IS_WINDOWS else "bin") / ("skill-hub-cli.exe" if IS_WINDOWS else "skill-hub-cli")
DEBUG_LOG = Path.home() / ".claude" / "mcp-skill-hub" / "logs" / "hook-debug.log"

# Close-task intent detection
_CLOSE_TASK_RE = _re.compile(
    r'\b(close|done with|finish(?:ed)?|wrap(?:ping)? up|mark.*done|complete(?:d)?)\b'
    r'.*\b(this\s+)?task\b',
    _re.IGNORECASE
)

_FALSE_POSITIVE_RE = _re.compile(
    r'\btask\b.*\bin\s+the\s+\w+|close\s+the\s+file|close\s+connection',
    _re.IGNORECASE
)

_t0 = time.monotonic()


def log(msg: str):
    try:
        DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        elapsed = time.monotonic() - _t0
        ts = datetime.now().strftime("%H:%M:%S")
        with open(DEBUG_LOG, "a") as f:
            f.write(f"[{ts}] [{elapsed:6.1f}s] INTERCEPT  {msg}\n")
    except OSError:
        pass


def _maybe_close_task(message: str, session_id: str) -> str:
    """Check if the user intends to close the current task.

    Returns an informational string if a task was closed, or "" if not.
    Never raises.
    """
    if not _CLOSE_TASK_RE.search(message):
        return ""
    if _FALSE_POSITIVE_RE.search(message):
        return ""
    try:
        sys.path.insert(0, str(SCRIPT_DIR / "src"))
        from skill_hub.store import SkillStore
        store = SkillStore()
        try:
            task_row = store.get_open_task_for_session(session_id)
            if not task_row:
                return ""
            task = dict(task_row)
            task_id = int(task["id"])
            # Use stored task title + summary as compact, not the raw close message
            compact_text = f"Closed via intent. Summary: {task.get('title', '')} — {task.get('summary', '')[:150]}"
            closed = store.close_task(task_id, compact=compact_text)
            if closed:
                info = f"Task #{task_id} closed."
                log(f"CLOSE_TASK  task_id={task_id}  compact_len={len(compact_text)}")
                return info
            return ""
        finally:
            store.close()
    except Exception as exc:
        log(f"close_task_intent  error={exc}")
        return ""


def warmup_ollama():
    """Pre-warm Ollama embed model (fire-and-forget)."""
    try:
        subprocess.Popen(
            [
                "curl", "-s", "--max-time", "2",
                "http://localhost:11434/api/embed",
                "-d", '{"model":"nomic-embed-text","input":"warmup","keep_alive":"10m"}',
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass


def main():
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return

    message = data.get("prompt", "") or data.get("userMessage", "")
    message_raw = message
    # Normalize whitespace for CLI arg safety
    message = message.replace("\n", " ").replace("\r", "").replace("\t", " ")
    session_id = data.get("session_id", "")

    # Preview: first 80 chars of user message for log readability
    preview = message_raw[:80].replace("\n", " ")
    if len(message_raw) > 80:
        preview += "..."

    log(f"fired  session={session_id}  len={len(message_raw)}  msg=\"{preview}\"")

    if not message:
        log("skip  reason=empty_message")
        return

    # Close-task intent — handle before LLM classification (0 tokens)
    close_info = _maybe_close_task(message, session_id)
    if close_info:
        print(close_info)

    warmup_ollama()

    # Classify via CLI
    cmd = [str(CLI), "classify"]
    if session_id:
        cmd.extend(["--session-id", session_id])
    cmd.append(message)

    t_cli = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=45,
        )
    except subprocess.TimeoutExpired:
        log("error  CLI timed out after 45s")
        return
    except OSError as e:
        log(f"error  CLI failed: {e}")
        return

    cli_ms = int((time.monotonic() - t_cli) * 1000)
    log(f"cli  exit={result.returncode}  time={cli_ms}ms  stdout_len={len(result.stdout)}")

    if result.returncode != 0 or not result.stdout.strip():
        if result.stderr:
            log(f"cli_stderr  {result.stderr.strip()[:200]}")
        log("allow  reason=cli_failed_or_empty")
        return

    try:
        cli_data = json.loads(result.stdout)
    except json.JSONDecodeError:
        log(f"error  invalid JSON: {result.stdout[:200]}")
        return

    decision = cli_data.get("decision", "allow")

    if decision == "block":
        blocked_msg = cli_data.get("message", "")[:100]
        log(f"BLOCK  msg=\"{blocked_msg}\"  cli_time={cli_ms}ms")
        # UserPromptSubmit block: use `continue: false` + `stopReason` (not `decision`).
        output = {
            "continue": False,
            "stopReason": cli_data.get("message") or cli_data.get("reason") or "Command handled locally.",
        }
        print(json.dumps(output))
    elif decision == "allow":
        has_system = bool(cli_data.get("systemMessage"))
        has_user = bool(cli_data.get("userMessage"))
        sys_len = len(cli_data.get("systemMessage", ""))
        log(f"ALLOW  enriched={'yes' if has_system else 'no'}  systemMsg={sys_len}chars  cli_time={cli_ms}ms")
        # Translate CLI's internal shape into Claude Code's hook-output schema.
        # `decision: "allow"` and top-level `userMessage` are NOT valid fields.
        output: dict = {}
        if has_system:
            output["systemMessage"] = cli_data["systemMessage"]
        if has_user:
            output["hookSpecificOutput"] = {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": cli_data["userMessage"],
            }
        if output:
            print(json.dumps(output))
    else:
        log(f"UNKNOWN decision={decision}  cli_time={cli_ms}ms")

    total_ms = int((time.monotonic() - _t0) * 1000)
    log(f"done  total_time={total_ms}ms")


if __name__ == "__main__":
    main()
