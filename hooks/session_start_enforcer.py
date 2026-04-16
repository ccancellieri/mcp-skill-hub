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


def _read_session_memory(session_id: str) -> str:
    """Return the stored 6-section session memory for this id, or empty.

    Best-effort: never raises; silently returns "" if skill-hub isn't
    importable or the file is missing. Truncated to ``session_memory_inject_max_chars``.
    """
    if not session_id:
        return ""
    try:
        from skill_hub import config as _cfg
        from skill_hub.router import session_memory as _sm
    except Exception:
        return ""
    if not _cfg.get("session_memory_inject_on_resume"):
        return ""
    text = _sm.read_memory(session_id)
    if not text.strip():
        return ""
    cap = int(_cfg.get("session_memory_inject_max_chars") or 8000)
    if len(text) > cap:
        text = text[:cap] + "\n\n<!-- session memory truncated -->"
    return (
        "RESUMED SESSION MEMORY (from previous turns — survives /compact):\n"
        "---\n"
        f"{text}\n"
        "---\n"
        "Use this as authoritative context for where you left off."
    )


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


def _ensure_open_tasks() -> str:
    """Scan MEMORY.md for PARTIAL/DEFERRED/ongoing entries that have no open task.

    Creates open tasks directly (Python API, no LLM cost) for any work that
    is tracked in memory but missing from the task list. Returns a brief
    advisory string listing created tasks, or empty string on no-op / error.

    Opt-out via config ``task_auto_create_enabled`` (default True).
    """
    try:
        from skill_hub import config as _cfg
        if _cfg.get("task_auto_create_enabled") is False:
            return ""
    except Exception:
        pass

    try:
        from skill_hub.store import SkillStore
        import re

        memory_index = Path.home() / ".claude" / "projects" / "-Users-ccancellieri-work-code" / "memory" / "MEMORY.md"
        # Fall back to scanning memory dirs that might exist
        if not memory_index.exists():
            memory_dirs = list(Path.home().glob(".claude/projects/*/memory/MEMORY.md"))
            if not memory_dirs:
                return ""
            memory_index = memory_dirs[0]

        text = memory_index.read_text(encoding="utf-8", errors="replace")

        # Patterns that indicate unfinished work
        partial_pattern = re.compile(
            r"PARTIAL|DEFERRED|WIP|ongoing|in.progress|wave-\d+ files remain|not yet|remain",
            re.IGNORECASE,
        )
        # Extract lines with links: - [title](file.md) — description
        link_re = re.compile(r"^[-*]\s+\[([^\]]+)\]\([^)]+\)\s*[—–-]+\s*(.+)$", re.MULTILINE)

        candidates: list[tuple[str, str]] = []  # (title, description)
        for m in link_re.finditer(text):
            title, desc = m.group(1).strip(), m.group(2).strip()
            if partial_pattern.search(desc):
                candidates.append((title, desc))

        if not candidates:
            return ""

        store = SkillStore()
        # Get all open task titles (lower-cased for fuzzy dedup)
        open_rows = store.list_tasks(status="open")
        open_titles_lower = {dict(r)["title"].lower() for r in open_rows}

        created: list[str] = []
        for title, desc in candidates:
            # Skip if a task with a very similar title already exists (substring match)
            title_lower = title.lower()
            already_exists = any(
                title_lower in existing or existing in title_lower
                for existing in open_titles_lower
            )
            if already_exists:
                continue

            # Derive tags from description keywords
            tags_raw = []
            for word in re.findall(r"\b(geoid|dynastore|mcp-skill-hub|pyright|duckdb|iceberg|ogc|stac|airflow|fao)\b", desc, re.IGNORECASE):
                tags_raw.append(word.lower())
            tags = ",".join(dict.fromkeys(tags_raw))  # dedup, preserve order

            store.save_task(
                title=title,
                summary=f"[auto-created from memory index]\n{desc}",
                vector=[],
                context="",
                tags=tags,
                session_id="",
            )
            open_titles_lower.add(title_lower)
            created.append(title)

        store.close()

        if not created:
            return ""
        names = "; ".join(f'"{t}"' for t in created[:3])
        extra = f" (+{len(created)-3} more)" if len(created) > 3 else ""
        return f"AUTO-TASKS: created {len(created)} open task(s) from memory — {names}{extra}. Visit /tasks to review."

    except Exception as exc:
        try:
            log(f"ensure_open_tasks error: {exc}")
        except Exception:
            pass
        return ""


def _auto_switch_profile() -> str:
    """Score profiles by tag overlap with the last N closed tasks and switch
    to the best match (when it differs from the current active profile).

    Opt-in via config ``profile_auto_switch_enabled`` (default False). The
    switch writes ``~/.claude/settings.json`` but only takes effect on the
    NEXT session — Claude Code has already loaded enabledPlugins by the time
    this hook runs. Returns an advisory string when a switch occurred, or
    empty string when no-op / disabled / nothing to do.
    """
    try:
        from skill_hub import config as _cfg
        from skill_hub.store import SkillStore
        from skill_hub import profiles as _prof
    except Exception:
        return ""
    if not _cfg.get("profile_auto_switch_enabled"):
        return ""
    try:
        store = SkillStore()
        window = int(_cfg.get("profile_auto_switch_window") or 5)
        rows = store._conn.execute(
            "SELECT tags FROM tasks WHERE status = 'closed' "
            "ORDER BY updated_at DESC LIMIT ?",
            (window,),
        ).fetchall()
        if not rows:
            return ""
        tag_counts: dict[str, int] = {}
        for r in rows:
            raw = (r["tags"] or "").strip()
            if not raw:
                continue
            for tag in (t.strip().lower() for t in raw.split(",")):
                if tag:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
        if not tag_counts:
            return ""
        all_profiles = _prof.list_profiles(store)
        best_name, best_score = "", 0
        for p in all_profiles:
            score = tag_counts.get(p["name"].lower(), 0)
            if score > best_score:
                best_name, best_score = p["name"], score
        if not best_name or best_score < 2:
            return ""
        active = _prof.get_active_profile(store)
        if active and active["name"] == best_name:
            return ""
        result = _prof.switch_profile(store, best_name)
        if not result.get("changed_plugins"):
            return ""
        return (
            f"AUTO-SWITCHED profile → {best_name!r} "
            f"(matched {best_score}/{len(rows)} recent task tags). "
            "Takes effect on next session restart."
        )
    except Exception:
        return ""


def _update_task_activity(session_id: str) -> None:
    """Update last_activity_at for the session's open task. Never raises."""
    if not session_id:
        return
    try:
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
        try:
            log(f"heartbeat  error={exc}")
        except Exception:
            pass


def _dispatch_background_jobs(session_id: str, ts: float) -> str:
    """Return a housekeeping block if background jobs are pending and user is idle.

    Returns empty string when disabled, no jobs, or user not yet idle enough.
    Never raises.
    """
    try:
        from skill_hub import config as _cfg
        if not _cfg.get("background_via_subagent_enabled"):
            return ""
    except Exception:
        return ""

    try:
        from skill_hub import background_jobs as _bj
        from skill_hub.store import SkillStore

        db_path = SkillStore().db_path  # type: ignore[attr-defined]
    except Exception:
        # Fallback: derive db_path the same way SkillStore does
        try:
            db_path = str(
                Path.home() / ".claude" / "mcp-skill-hub" / "skill_hub.db"
            )
        except Exception:
            return ""

    try:
        if _bj.get_pending_count(db_path) == 0:
            return ""

        try:
            idle_ms = int(
                _cfg.get("background_subagent_idle_threshold_ms") or 3000
            )
        except Exception:
            idle_ms = 3000

        if not _bj.should_dispatch(ts, idle_threshold_ms=idle_ms):
            return ""

        try:
            max_jobs = int(_cfg.get("background_max_jobs_per_prompt") or 1)
        except Exception:
            max_jobs = 1

        jobs = _bj.list_pending_jobs(db_path, max_jobs=max_jobs)
        if not jobs:
            return ""
        return _bj.build_housekeeping_block(jobs)
    except Exception:
        return ""


def log(msg: str):
    try:
        DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%H:%M:%S")
        with open(DEBUG_LOG, "a") as f:
            tag = verdict_cache.task_tag()
            f.write(f"[{ts}] [  0.0s] ENFORCER   {tag}{msg}\n")
    except OSError:
        pass


def _run_pipeline(message: str, session_id: str) -> str:
    """Run the 4-tier pre-conversation pipeline if enabled.

    Returns a systemMessage block with synthesis + task_id, or empty string.
    Never raises.
    """
    try:
        from skill_hub import config as _cfg
        if not _cfg.get("pre_conversation_pipeline_enabled"):
            return ""
    except Exception:
        return ""

    try:
        from skill_hub.pipeline import Pipeline
        pipe = Pipeline()
        result = pipe.run(message=message, session_id=session_id)

        parts = []
        if result.task_id:
            parts.append(f"[task #{result.task_id} tracking this session]")
        if result.synthesis:
            parts.append(f"Context from prior work: {result.synthesis}")
        if result.enriched_prompt:
            parts.append(f"Suggested refined prompt: {result.enriched_prompt}")

        if not parts:
            return ""

        return "PIPELINE CONTEXT:\n" + "\n".join(parts)
    except Exception as exc:
        try:
            log(f"pipeline error: {exc}")
        except Exception:
            pass
        return ""


def _maybe_teach_from_message(message: str, session_id: str) -> str:
    """If first message contains a teach-directive, auto-teach it. Returns advisory."""
    if not message.strip():
        return ""
    try:
        from skill_hub import config as _cfg
        if not _cfg.get("continuous_teaching_enabled"):
            return ""
    except Exception:
        return ""

    import re
    teach_patterns = [
        r'^(?:please\s+)?remember[:\s]+(.{10,300})$',
        r'^never (?:do |again )?(?:this[:\s]+)?(.{10,300})$',
        r'^always (?:do )?(?:this[:\s]+)?(.{10,300})$',
        r'(?:please\s+)?remember[:\s]+(.{10,200})(?:going forward|from now on|always)',
    ]
    for pat in teach_patterns:
        m = re.search(pat, message.strip(), re.IGNORECASE | re.DOTALL)
        if m:
            rule_text = m.group(1).strip()
            try:
                from skill_hub.store import SkillStore as _SK
                _store = _SK()
                try:
                    try:
                        from skill_hub.embeddings import embed as _embed
                        vec = _embed(rule_text)
                    except Exception:
                        vec = []
                    _store.add_teaching(
                        rule=rule_text, rule_vector=vec,
                        action="Auto-taught from session start message",
                        target_type="global", target_id="global",
                    )
                    return f'AUTO-TAUGHT: "{rule_text[:80]}"'
                finally:
                    _store.close()
            except Exception:
                pass
    return ""


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

    memory_msg = _read_session_memory(session_id)
    if memory_msg:
        log(f"SESMEM injected  chars={len(memory_msg)}")

    auto_switch_msg = _auto_switch_profile()
    if auto_switch_msg:
        log(f"PROFILE auto-switch  msg=\"{auto_switch_msg[:100]}\"")

    drift_msg = _check_profile_drift()
    if drift_msg:
        log(f"PROFILE drift detected  msg=\"{drift_msg[:100]}\"")

    tasks_msg = _ensure_open_tasks()
    if tasks_msg:
        log(f"AUTO-TASKS  msg=\"{tasks_msg[:120]}\"")

    # Heartbeat: update last_activity_at for the session's open task
    _update_task_activity(session_id)

    user_message = data.get("message") or data.get("prompt") or ""

    # Auto-teach from "remember X" / "never do X" / "always do X" in first message
    teach_advisory = _maybe_teach_from_message(user_message, session_id)
    if teach_advisory:
        log(f"AUTO-TEACH  msg=\"{teach_advisory[:100]}\"")

    pipeline_msg = _run_pipeline(user_message, session_id)
    if pipeline_msg:
        log(f"PIPELINE  chars={len(pipeline_msg)}")

    import time as _time
    housekeeping_msg = _dispatch_background_jobs(session_id, _time.time())
    if housekeeping_msg:
        log(f"HOUSEKEEPING  chars={len(housekeeping_msg)}")

    log(f"injecting session-start reminder  log_cmd=\"{log_cmd}\"")

    system_msg = (
        "SESSION START:\n"
        "1. Read project .memory/index.md if it exists in the working directory\n"
        "2. Follow CLAUDE.md multi-level context protocol\n"
        "\n"
        f"Hook activity log: {log_cmd}\n"
        "Mention the log command to the user so they can follow local LLM activity."
    )
    if pipeline_msg:
        system_msg = pipeline_msg + "\n\n" + system_msg
    if memory_msg:
        system_msg = memory_msg + "\n\n" + system_msg
    if resume_msg:
        system_msg = resume_msg + "\n\n" + system_msg
    if auto_switch_msg:
        system_msg = auto_switch_msg + "\n\n" + system_msg
    if drift_msg:
        system_msg = drift_msg + "\n\n" + system_msg
    if tasks_msg:
        system_msg = tasks_msg + "\n\n" + system_msg
    if teach_advisory:
        system_msg = teach_advisory + "\n\n" + system_msg
    if housekeeping_msg:
        system_msg = system_msg + "\n\n" + housekeeping_msg

    print(json.dumps({"decision": "allow", "systemMessage": system_msg}))


if __name__ == "__main__":
    main()
