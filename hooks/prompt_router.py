#!/usr/bin/env python3
"""UserPromptSubmit hook: prompt router (cross-platform).

Flow:
  1. Read user prompt from stdin JSON
  2. Call skill-hub-cli route --session-id <id> <cwd> <message>
  3. Emit the resulting systemMessage / userMessage to stdout
  4. Claude Code picks them up before Claude sees the prompt

The hook adds no latency on Tier-1 paths (<10ms).
Tier-2 (Ollama) adds ~200-500ms; Tier-3 (Haiku) ~500ms. Both are gated
by confidence thresholds so they only fire when actually uncertain.
"""

import json
import os
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

_t0 = time.monotonic()


def log(msg: str) -> None:
    try:
        DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        elapsed = time.monotonic() - _t0
        ts = datetime.now().strftime("%H:%M:%S")
        with open(DEBUG_LOG, "a") as f:
            f.write(f"[{ts}] [{elapsed:6.1f}s] ROUTER     {msg}\n")
    except OSError:
        pass


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return

    message = data.get("prompt", "") or data.get("userMessage", "")
    message = message.replace("\n", " ").replace("\r", "").replace("\t", " ")
    session_id = data.get("session_id", "")
    cwd = data.get("cwd", os.getcwd())

    preview = message[:80] + ("..." if len(message) > 80 else "")
    log(f"fired  session={session_id}  len={len(message)}  msg=\"{preview}\"")

    if not message:
        log("skip  reason=empty_message")
        return

    # Read active task marker — provides task_id for telemetry and option overrides.
    _active_marker = Path.home() / ".claude" / "mcp-skill-hub" / "state" / "active_task.json"
    active_task: dict = {}
    try:
        if _active_marker.exists():
            active_task = json.loads(_active_marker.read_text()) or {}
    except (OSError, json.JSONDecodeError):
        pass

    # Check per-task routing_disabled option — skip routing entirely if set.
    task_options: dict = active_task.get("options") or {}
    if task_options.get("routing_disabled"):
        log(f"skip  reason=routing_disabled  task_id={active_task.get('task_id')}")
        return

    active_task_id = active_task.get("task_id")

    cmd = [str(CLI), "route"]
    if session_id:
        cmd.extend(["--session-id", session_id])
    if cwd:
        cmd.extend(["--cwd", cwd])
    if active_task_id is not None:
        cmd.extend(["--task-id", str(active_task_id)])
    cmd.append(message)

    t_cli = time.monotonic()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        log("error  route CLI timed out after 20s")
        return
    except OSError as e:
        log(f"error  route CLI failed: {e}")
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

    if not cli_data:
        log("allow  reason=no_verdict")
        return

    # Translate CLI's internal shape (decision/userMessage) into Claude Code's
    # hook-output schema: top-level systemMessage + hookSpecificOutput.additionalContext.
    # `decision: "allow"` and top-level `userMessage` are NOT valid fields.
    # Strip privileged tags that Claude Code's hook-output validator rejects
    # as prompt injection (produces "Hook JSON output validation failed —
    # (root): Invalid input"). Covers user-authored task titles that happened
    # to capture a <system-reminder> block.
    import re as _re
    _PRIV_TAG = _re.compile(
        r"</?\s*(?:system-reminder|system|assistant|user|tool_use|tool_result|"
        r"function_calls|antml:[a-z_]+)\s*/?>",
        _re.IGNORECASE,
    )

    def _sanitize(s: str) -> str:
        if not s:
            return ""
        s = _PRIV_TAG.sub("", s)
        s = "".join(ch for ch in s if ch in ("\t", "\n") or ord(ch) >= 0x20)
        return s

    output: dict = {}
    if cli_data.get("systemMessage"):
        output["systemMessage"] = _sanitize(cli_data["systemMessage"])
    if cli_data.get("userMessage"):
        output["hookSpecificOutput"] = {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": _sanitize(cli_data["userMessage"]),
        }

    if output:
        log(f"VERDICT  sys={bool(output.get('systemMessage'))}  user={bool(output.get('hookSpecificOutput'))}  cli_time={cli_ms}ms")
        print(json.dumps(output))
    else:
        log("allow  reason=empty_verdict")


if __name__ == "__main__":
    main()
