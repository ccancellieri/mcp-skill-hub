#!/usr/bin/env python3
"""PostToolUse / PostToolUseFailure hook: record tool outcomes as training signal.

PostToolUse: if Claude ran a Bash command successfully, the user either
approved it or it was auto-approved. Either way the command was judged
safe-enough in this project context — record it so future identical/similar
commands can be auto-approved from cache without re-prompting.

PostToolUseFailure (Claude Code 2.1.119): the same Bash command failed.
We record it as ``status=failed`` so the verdict cache learns *not* to
auto-approve flaky commands. ``duration_ms`` from the hook input is also
captured for telemetry on slow runs.

Only writes to cache when:
  - tool_name is Bash (others are handled by safe_tools list)
  - PostToolUse: the tool did not error (we re-check tool_response too because
    the success/failure split between events isn't always honoured by hosts)
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


def _update_task_activity(session_id: str) -> None:
    """Update last_activity_at for the session's open task. Never raises."""
    if not session_id:
        return
    try:
        import sys
        from pathlib import Path as _Path
        sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "src"))
        from skill_hub.store import SkillStore
        store = SkillStore()
        try:
            task_id = store.get_open_task_id_for_session(session_id)
            if task_id:
                store.touch_task_activity(task_id)
                log(f"HEARTBEAT  task_id={task_id}")
        finally:
            store.close()
    except Exception as exc:
        log(f"heartbeat  error={exc}")


def _maybe_auto_teach_from_feedback(tool_name: str, tool_input: dict) -> None:
    """If a Write/Edit touched a feedback_*.md, auto-teach from it. Best-effort."""
    if tool_name not in ("Write", "Edit"):
        return

    file_path = tool_input.get("file_path") or tool_input.get("path") or ""
    import re
    if not re.search(r'feedback_[^/\\]+\.md$', file_path):
        return

    try:
        from skill_hub import config as _cfg
        if not _cfg.get("continuous_teaching_enabled"):
            return
    except Exception:
        return

    try:
        from pathlib import Path
        content = Path(file_path).read_text(encoding="utf-8", errors="replace")
        rule = ""
        why = ""

        # Strip YAML frontmatter
        fm_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
        if fm_match:
            body = content[fm_match.end():]
        else:
            body = content

        # Get first paragraph as rule
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', body) if p.strip()]
        if paragraphs:
            rule = paragraphs[0][:500]

        # Find **Why:** section
        why_match = re.search(r'\*\*Why:\*\*\s*(.+?)(?=\n\n|\Z)', body, re.DOTALL)
        if why_match:
            why = why_match.group(1).strip()[:300]

        if not rule:
            return

        from skill_hub.store import SkillStore
        store = SkillStore()
        try:
            try:
                from skill_hub.embeddings import embed as _embed
                vec = _embed(rule)
            except Exception:
                vec = []

            store.add_teaching(
                rule=rule,
                rule_vector=vec,
                action=why or "Auto-taught from feedback file update",
                target_type="global",
                target_id="global",
            )
            log(f"auto-teach  feedback_file=\"{file_path[-60:]}\"  rule=\"{rule[:60]}\"")
        finally:
            store.close()

    except Exception as e:  # noqa: BLE001
        log(f"auto-teach  error={e}")


def main() -> int:
    cfg = verdict_cache.load_config()
    if not cfg.get("auto_approve_learn", True):
        return 0

    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return 0

    session_id = data.get("session_id", "")
    _update_task_activity(session_id)

    event = data.get("hook_event_name", "PostToolUse")
    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input") or {}
    duration_ms = data.get("duration_ms")

    # Phase G.2 — auto-teach from feedback file writes (always attempt, regardless
    # of auto_approve_learn; continuous_teaching_enabled guards it internally).
    # Failure events skip auto-teach: a feedback file written then immediately
    # erroring out shouldn't propagate as a teaching.
    if event == "PostToolUse":
        _maybe_auto_teach_from_feedback(tool_name, tool_input)

    if tool_name != "Bash":
        return 0

    cmd = (tool_input.get("command") or "").strip()
    if not cmd:
        return 0

    if event == "PostToolUseFailure":
        # Negative reinforcement: store as a `failed` verdict so the verdict
        # cache lookup can downweight or refuse to auto-approve repeat runs.
        try:
            conn = verdict_cache.connect()
            verdict_cache.put(conn, tool_name, cmd, "failed", "tool_failure", 1.0)
            log(f"failed  cmd=\"{cmd[:60]}\"  duration_ms={duration_ms}")
        except Exception as e:  # noqa: BLE001
            log(f"cache  error={e}")
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
        log(f"learned  cmd=\"{cmd[:60]}\"  duration_ms={duration_ms}")
    except Exception as e:  # noqa: BLE001
        log(f"cache  error={e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
