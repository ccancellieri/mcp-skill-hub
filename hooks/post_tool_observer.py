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
import os
import sys
from datetime import datetime
from pathlib import Path

_HOOK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_HOOK_DIR))
sys.path.insert(0, str(_HOOK_DIR.parent / "src"))
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


def _maybe_observe_claude_task(data: dict) -> None:
    """Project Claude Code task tool calls into skill-hub tasks. Never raises."""
    try:
        from skill_hub import claude_tasks as _ct
        tool_name = data.get("tool_name", "")
        if tool_name not in _ct.CLAUDE_TASK_TOOLS:
            return
        tool_input = data.get("tool_input") or {}
        tool_response = data.get("tool_response") or {}
        if isinstance(tool_response, str):
            tool_response = {}
        parsed = _ct.parse_claude_tasks(tool_name, tool_input, tool_response)
        if not parsed:
            return
        session_id = data.get("session_id", "")
        cwd = data.get("cwd") or os.getcwd()
        branch = ""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, cwd=cwd, timeout=2,
            )
            if result.returncode == 0:
                branch = result.stdout.strip()
        except Exception:
            pass

        from skill_hub.store import SkillStore
        store = SkillStore()
        try:
            for item in parsed:
                key = _ct.stable_key(
                    item["identity"],
                    cwd=cwd,
                    branch=branch,
                    claude_id=item.get("claude_id"),
                )
                status = item["status"]
                is_completion = status in _ct._COMPLETION_STATUSES
                event_kind = (
                    "claude_task.completed" if is_completion else "claude_task.observed"
                )
                payload = {
                    "key": key,
                    "title": item["title"],
                    "status": status,
                    "claude_id": item.get("claude_id"),
                    "tool_name": tool_name,
                }
                store.append_event(
                    session_id, event_kind, tool_name, payload
                )
                result_info = store.project_claude_task(
                    key=key,
                    title=item["title"],
                    status=status,
                    claude_id=item.get("claude_id"),
                    session_id=session_id,
                    cwd=cwd,
                    branch=branch,
                    summary="",
                )
                log(
                    f"claude_task  action={result_info.get('action')}  "
                    f"key={key[:20]}  title=\"{item['title'][:40]}\""
                )
        finally:
            store.close()
    except Exception as exc:  # noqa: BLE001
        log(f"claude_task  error={exc}")


_SEARCH_SKILLS_TOOL = "mcp__skill-hub__search_skills"
# Regex to extract "LOADED (N): id1, id2, ..." from search_skills response text.
import re as _re
_LOADED_RE = _re.compile(r"<!-- LOADED \(\d+\):\s*(.*?)-->")


def _maybe_emit_skill_used(
    tool_name: str, tool_response: object, session_id: str
) -> None:
    """Emit ``skill.used`` events when search_skills returns skill content.

    Fires on PostToolUse for the search_skills MCP tool.  Parses the
    ``LOADED (N): id1, id2`` header from the response text to identify
    which skills were served, then calls ``store.record_skill_used`` for
    each one (matched to the most-recent injection row for that
    skill+session).  Never raises.
    """
    if tool_name != _SEARCH_SKILLS_TOOL:
        return

    # tool_response for MCP text tools is a string in the hook payload.
    if isinstance(tool_response, dict):
        response_text = (
            tool_response.get("content") or tool_response.get("text") or ""
        )
        if isinstance(response_text, list):
            # content may be [{"type":"text","text":"..."}]
            response_text = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in response_text
            )
    elif isinstance(tool_response, str):
        response_text = tool_response
    else:
        return

    m = _LOADED_RE.search(response_text)
    if not m:
        return

    raw_ids = m.group(1).strip()
    if not raw_ids or raw_ids == "none":
        return
    skill_ids = [s.strip() for s in raw_ids.split(",") if s.strip() and s.strip() != "none"]
    if not skill_ids:
        return

    try:
        from pathlib import Path as _Path
        import sys as _sys
        _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "src"))
        from skill_hub.store import SkillStore
        store = SkillStore()
        try:
            for sid in skill_ids:
                store.record_skill_used(sid, session_id)
                log(f"skill.used  skill_id={sid}  session={session_id[:12]}")
        finally:
            store.close()
    except Exception as exc:  # noqa: BLE001
        log(f"skill.used  error={exc}")


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


# ── Tool-usage activity logging (issue: codegraph-vs-grep visibility) ──
# The activity log only ever showed Bash commands (grep), never the MCP tools
# Claude actually ran (codegraph_search, search_skills, ...). PostToolUse is the
# one hook that sees *every* tool call, so we surface a compact `TOOL` line here
# — and flag grep/rg used inside a `.codegraph/`-indexed repo, where
# codegraph_search would be the better move.

_GREP_CMD_RE = _re.compile(
    r"^\s*(?:[\w./-]*/)?(?:grep|egrep|fgrep|rg|ag|ack)\b"
)
_GREP_PIPE_RE = _re.compile(r"\|\s*(?:grep|egrep|fgrep|rg|ag|ack)\b")


def _mcp_short(tool_name: str) -> str:
    """`mcp__codegraph__codegraph_search` -> `codegraph:search`."""
    parts = tool_name.split("__")
    if len(parts) >= 3:
        server, tool = parts[1], "__".join(parts[2:])
        if tool.startswith(server + "_"):
            tool = tool[len(server) + 1:]
        return f"{server}:{tool}"
    return tool_name


def _describe_tool(tool_name: str, tool_input: dict) -> tuple[str, str, bool]:
    """Return (display_name, detail, is_greplike) for one tool call."""
    ti = tool_input or {}
    if tool_name == "Bash":
        raw = (ti.get("command") or "").strip()
        cmd = raw.splitlines()[0] if raw else ""
        greplike = bool(_GREP_CMD_RE.match(cmd) or _GREP_PIPE_RE.search(cmd))
        return "Bash", cmd[:80], greplike
    if tool_name in ("Read", "Write", "Edit", "NotebookEdit"):
        path = str(ti.get("file_path") or ti.get("notebook_path") or "")
        return tool_name, path[-70:], False
    if tool_name == "Grep":
        pat = str(ti.get("pattern") or "")
        where = str(ti.get("path") or ti.get("glob") or "")
        return "Grep", f"{pat}  {where}".strip()[:80], True
    if tool_name == "Glob":
        return "Glob", str(ti.get("pattern") or "")[:80], False
    if tool_name in ("Task", "Agent"):
        detail = f"{ti.get('subagent_type', '')} {ti.get('description', '')}".strip()
        return tool_name, detail[:80], False
    if tool_name.startswith("mcp__"):
        key = (ti.get("query") or ti.get("symbol") or ti.get("name")
               or ti.get("pattern") or ti.get("path") or "")
        return _mcp_short(tool_name), str(key)[:80], False
    return tool_name, "", False


def _codegraph_indexed(cwd: str) -> bool:
    """True if `cwd` (or any parent) holds a `.codegraph/` index."""
    try:
        p = Path(cwd or os.getcwd()).resolve()
        for d in (p, *p.parents):
            if (d / ".codegraph").is_dir():
                return True
    except Exception:
        pass
    return False


def _emit_tool_activity(tool_name: str, tool_input: dict, cwd: str,
                        cfg: dict) -> None:
    """Write a compact `TOOL` line (and a codegraph `HINT`) to activity.log."""
    if not tool_name or not cfg.get("log_tool_usage", True):
        return
    try:
        from skill_hub.activity_log import append_line
    except Exception:
        return
    name, detail, greplike = _describe_tool(tool_name, tool_input)
    append_line(f"TOOL  {name:<22}{detail}".rstrip())
    if (greplike and cfg.get("tool_usage_codegraph_hint", True)
            and _codegraph_indexed(cwd)):
        append_line("HINT  codegraph indexed here — prefer codegraph_search")


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
        _emit_tool_activity(tool_name, tool_input, data.get("cwd") or "", cfg)
        _maybe_auto_teach_from_feedback(tool_name, tool_input)
        _maybe_observe_claude_task(data)
        _maybe_emit_skill_used(tool_name, data.get("tool_response"), session_id)

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
