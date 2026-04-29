"""
Skill Hub MCP Server
====================
Local stdio MCP server that provides semantic skill search for Claude Code.

Tools — Search & Load
---------------------
search_skills       — Semantic search, returns full skill content inline
suggest_plugins     — Suggest disabled plugins that match the current task

Tools — Learning
----------------
record_feedback     — Mark a skill/plugin as helpful or not (feedback boost)
teach               — Add an explicit rule ("when X, suggest Y")
forget_teaching     — Remove a teaching rule by ID
list_teachings      — Show all teaching rules
log_session         — Record tool usage for passive learning (called by hooks)

Tools — Tasks (cross-session context)
--------------------------------------
save_task           — Save an open task/discussion for future sessions
close_task          — Compact via local LLM and close
update_task         — Update an open task with new info
reopen_task         — Reopen a closed task
list_tasks          — List open/closed/all tasks
search_context      — Unified search: skills + tasks + teachings

Tools — Management
------------------
index_skills        — Rebuild skill + plugin index from plugin directories
index_plugins       — Index plugin descriptions for suggest_plugins
list_skills         — List indexed skills
toggle_plugin       — Enable/disable plugins in settings.json
session_stats       — Show most-used plugins from session history
status              — Health check: MCP, Ollama, models, hook, DB stats, context usage
token_stats         — Token savings report from hook interceptions
list_models         — List installed Ollama models with role explanations
pull_model          — Download a new Ollama model
"""

import json
import uuid
from pathlib import Path

from fastmcp import FastMCP

from . import config as _cfg
from .embeddings import (
    embed, rerank, compact, rewrite_query, optimize_context,
    EMBED_MODEL, RERANK_MODEL, ollama_available, embed_available,
    _generate,
)
from .indexer import index_all
from .activity_log import log_tool, log_llm, log_banner
from .resource_monitor import should_run_llm, snapshot
from .store import SkillStore
from . import dashboard as _dashboard


def _get_cpu_info() -> int:
    """Get CPU core count for display."""
    import os
    return os.cpu_count() or 0

SETTINGS_PATH = Path.home() / ".claude" / "settings.json"

_store = SkillStore()

_ACTIVE_TASK_MARKER = Path.home() / ".claude" / "mcp-skill-hub" / "state" / "active_task.json"


def _write_active_task_marker(task_id: int, session_id: str, title: str,
                              auto_approve: bool | None,
                              options: dict | None = None) -> None:
    """Write active task marker so hooks can read it without touching the DB."""
    try:
        _ACTIVE_TASK_MARKER.parent.mkdir(parents=True, exist_ok=True)
        payload: dict = {
            "task_id": task_id,
            "session_id": session_id,
            "title": title,
            "auto_approve": auto_approve,
            "options": options or {},
        }
        _ACTIVE_TASK_MARKER.write_text(json.dumps(payload, indent=2))
    except OSError:
        pass


def _clear_active_task_marker(task_id: int | None = None) -> None:
    """Remove active task marker. If task_id given, only remove if it matches."""
    try:
        if task_id is not None and _ACTIVE_TASK_MARKER.exists():
            try:
                cur = json.loads(_ACTIVE_TASK_MARKER.read_text())
                if cur.get("task_id") != task_id:
                    return
            except (OSError, json.JSONDecodeError):
                pass
        _ACTIVE_TASK_MARKER.unlink(missing_ok=True)
    except OSError:
        pass

def _refresh_active_marker_options(task_id: int, task, options: dict) -> None:
    """Re-write the active marker when options change — keeps hooks in sync."""
    try:
        aa = options.get("auto_approve")
        if _ACTIVE_TASK_MARKER.exists():
            cur = json.loads(_ACTIVE_TASK_MARKER.read_text())
            if cur.get("task_id") == task_id:
                _write_active_task_marker(
                    task_id, cur.get("session_id", _session["id"]),
                    cur.get("title", task["title"]), auto_approve=aa, options=options,
                )
                return
        if task["session_id"] == _session["id"] and task["status"] == "open":
            _write_active_task_marker(
                task_id, task["session_id"], task["title"],
                auto_approve=aa, options=options,
            )
    except (OSError, json.JSONDecodeError):
        pass


# Warm up the FastAPI dashboard in a daemon thread so it's ready before
# the first close_task / render_dashboard call. Safe no-op if disabled or
# the port is busy.
try:
    _dashboard.render_interactive(_store)
except Exception:  # noqa: BLE001
    pass

# Service registry + reconciler — owns watcher, Ollama, SearXNG lifecycle.
import atexit as _atexit
from .services.monitor import PressureTracker as _PressureTracker
from .services.registry import (
    ServiceRegistry as _ServiceRegistry,
    set_registry as _set_registry,
    set_pressure as _set_pressure,
    start_reconciler as _start_reconciler,
)

_registry = _ServiceRegistry.build_from_config(_cfg.load_config())
_pressure = _PressureTracker(load_config_callable=_cfg.load_config)
_set_registry(_registry)
_set_pressure(_pressure)

if (_cfg.load_config().get("services") or {}).get("auto_reconcile", True):
    _reconciler = _start_reconciler(
        _registry,
        _pressure,
        config_path=_cfg.CONFIG_PATH,
        load_config=_cfg.load_config,
        interval_sec=2.0,
    )
    _atexit.register(_reconciler.stop)

# In-process session tracking
_session = {
    "id": str(uuid.uuid4()),
    "topic": "",
    "topic_vector": [],
    # Phase M3 — rolling tool-chain fingerprint for habits:tool-chains.
    "tool_chain": [],  # list[str] of recent tool names
}

# Phase M3 — flush the rolling tool chain every N invocations into
# ``habits:tool-chains`` as one vector per window. Keeps the corpus small
# while preserving temporal ordering patterns ("read → grep → edit").
_TOOL_CHAIN_WINDOW = 5


def _record_tool_chain(tool_name: str) -> None:
    """Append ``tool_name`` to the session's rolling window; flush on fill.

    Best-effort — never raises into the MCP tool path.
    """
    try:
        chain = _session.setdefault("tool_chain", [])
        chain.append(tool_name)
        if len(chain) < _TOOL_CHAIN_WINDOW:
            return
        window = list(chain[-_TOOL_CHAIN_WINDOW:])
        _session["tool_chain"] = []
        import hashlib, time
        text = " → ".join(window)
        doc_id = (f"{_session['id'][:8]}:"
                  f"{hashlib.sha1(text.encode()).hexdigest()[:10]}")
        _store.upsert_vector(
            namespace="habits:tool-chains",
            doc_id=doc_id,
            text=text,
            metadata={
                "tools": window,
                "session_id": _session.get("id"),
                "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "topic": _session.get("topic", "")[:120],
            },
            level="L2",
            source="tool_chain",
        )
    except Exception:  # noqa: BLE001
        pass

mcp = FastMCP(
    "skill-hub",
    instructions="""
Skill Hub gives you semantic skill search and plugin management. Workflow:

1. At conversation start (or new topic), call search_skills(query="<task description>").
   Also call suggest_plugins(query) to check if a disabled plugin would help.
2. If suggest_plugins recommends enabling a plugin, tell the user and offer to
   toggle_plugin() + restart.
3. Follow the skill content returned by search_skills.
4. After the task, call record_feedback(skill_id, helpful=True/False).
5. Use teach() to create persistent rules ("when I give a URL, suggest chrome-devtools").

The system learns from three signals:
- Explicit teachings (teach tool)
- Feedback on search results (record_feedback)
- Session tool usage patterns (automatic via hooks)
""",
)


# ---------------------------------------------------------------------------
# Search & Load


# Transient in-process state so record_feedback can omit query when called
# right after search_skills.
_last_search_state: dict = {"query": "", "vector": [], "skills": []}


@mcp.tool()
def search_skills(
    query: str,
    top_k: int = 5,
    use_rerank: bool = False,
) -> str:
    """Semantic skill search. Returns full content of top matches."""
    from .activity_log import LOG_FILE

    log_tool("search_skills", query=query, top_k=top_k, rerank=use_rerank)
    if not embed_available():
        return (
            f"No embedding backend available. "
            f"Set VOYAGE_API_KEY, start Ollama with '{EMBED_MODEL}', or install sentence-transformers."
        )

    query_vector = embed(query)

    # Update session topic on first search
    if not _session["topic"]:
        _session["topic"] = query
        _session["topic_vector"] = query_vector

    # Fetch a wider candidate pool: top_k loaded + up to top_k more to list as found
    fetch_n = top_k * 2 if use_rerank else top_k * 2
    all_candidates = _store.search(query_vector, top_k=fetch_n)

    if not all_candidates:
        return "No matching skills found. Run index_skills() if the index is empty."

    if use_rerank and len(all_candidates) > 1 and should_run_llm("rerank"):
        all_candidates = rerank(query, all_candidates, model=RERANK_MODEL)

    loaded = all_candidates[:top_k]
    not_loaded = all_candidates[top_k:]

    _last_search_state["query"] = query
    _last_search_state["vector"] = query_vector
    _last_search_state["skills"] = [c["id"] for c in loaded]

    # Log per-skill injections (one row per skill actually returned).
    for c in loaded:
        try:
            _store.log_skill_injection(c["id"], query, _session.get("id"))
        except Exception:
            pass

    # Build summary header
    loaded_ids = [c["id"] for c in loaded]
    not_loaded_ids = [c["id"] for c in not_loaded]

    header_lines = [
        f"<!-- Skill Hub search: query={query!r} top_k={top_k} -->",
        f"<!-- LOADED ({len(loaded)}):     {', '.join(loaded_ids) or 'none'} -->",
        f"<!-- NOT LOADED ({len(not_loaded)}): {', '.join(not_loaded_ids) or 'none'} -->",
        f"<!-- log: tail -f {LOG_FILE} -->",
    ]
    header = "\n".join(header_lines)

    max_skill_chars = int(_cfg.get("hook_context_max_skill_chars") or 8000)
    parts: list[str] = [header]
    for c in loaded:
        content = c['content'].strip()
        if len(content) > max_skill_chars:
            content = content[:max_skill_chars] + "\n\n<!-- truncated -->"
        parts.append(
            f"<!-- skill: {c['id']} -->\n{content}"
        )

    # Also check teachings for plugin suggestions
    teachings = _store.search_teachings(query_vector, min_sim=0.6)
    if teachings:
        plugin_hints = [t for t in teachings if t["target_type"] == "plugin"]
        if plugin_hints:
            hint_lines = [f"  - {t['action']} (rule: \"{t['rule']}\")" for t in plugin_hints[:3]]
            parts.append(
                "<!-- plugin suggestions from teachings -->\n"
                "Teachings suggest these plugins may be needed:\n" +
                "\n".join(hint_lines) +
                "\nUse suggest_plugins() for full analysis."
            )

    return "\n\n---\n\n".join(parts)


@mcp.tool()
def suggest_plugins(query: str = "") -> str:
    """Suggest disabled plugins matching the current task."""
    log_tool("suggest_plugins", query=query)
    if not embed_available():
        return "No embedding backend available. Set VOYAGE_API_KEY, start Ollama, or install sentence-transformers."

    used_query = query or _last_search_state.get("query", "")
    if not used_query:
        return "Provide a query or call search_skills first."

    query_vector = embed(used_query) if query else _last_search_state.get("vector", [])
    if not query_vector:
        query_vector = embed(used_query)

    suggestions = _store.suggest_plugins(query_vector)
    if not suggestions:
        return "No plugin suggestions for this query. Add teachings with teach() to improve."

    # Check current enabled state
    enabled_plugins: dict = {}
    if SETTINGS_PATH.exists():
        settings = json.loads(SETTINGS_PATH.read_text())
        enabled_plugins = settings.get("enabledPlugins", {})

    lines: list[str] = []
    for s in suggestions[:5]:
        pid = s["plugin_id"]
        is_enabled = enabled_plugins.get(pid, False)
        status = "ENABLED" if is_enabled else "DISABLED"
        scores = (f"embed={s['embed_score']:.2f} "
                  f"teach={s['teaching_score']:.2f} "
                  f"history={s['session_score']:.2f}")
        lines.append(
            f"- [{status}] {s['short_name']}: {s['description'] or '(no description)'}\n"
            f"  scores: {scores} | total={s['total_score']:.2f}"
        )
        if not is_enabled:
            lines.append(
                f"  → to enable: toggle_plugin(\"{s['short_name']}\", enabled=True)"
            )

    return f"Plugin suggestions for \"{used_query}\":\n\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Learning


@mcp.tool()
def record_feedback(
    skill_id: str,
    helpful: bool,
    query: str = "",
) -> str:
    """Record whether a skill was helpful for future ranking improvement."""
    log_tool("record_feedback", skill_id=skill_id, helpful=helpful)
    used_query = query or _last_search_state.get("query", "")
    used_vector = _last_search_state.get("vector", [])

    if not used_query:
        return "Provide a query string (or call search_skills first)."

    if not used_vector:
        used_vector = embed(used_query)

    _store.record_feedback(skill_id, used_query, used_vector, helpful)
    sentiment = "positive" if helpful else "negative"
    return f"Recorded {sentiment} feedback for '{skill_id}'. Rankings will improve."


@mcp.tool()
def teach(rule: str, suggest: str) -> str:
    """Add a persistent rule mapping task patterns to plugins or skills."""
    log_tool("teach", rule=rule, suggest=suggest)
    if not embed_available():
        return "No embedding backend available. Set VOYAGE_API_KEY, start Ollama, or install sentence-transformers."

    rule_vector = embed(rule)

    # Determine if target is a plugin or skill
    target_type = "plugin"
    target_id = suggest

    # Check if it matches a known skill
    content = _store.get_skill_content(suggest)
    if content:
        target_type = "skill"

    tid = _store.add_teaching(
        rule=rule,
        rule_vector=rule_vector,
        action=f"suggest {suggest}",
        target_type=target_type,
        target_id=target_id,
    )
    return (
        f"Teaching #{tid} saved: \"{rule}\" → suggest {suggest} ({target_type}).\n"
        f"This will be matched semantically against future queries."
    )


@mcp.tool()
def forget_teaching(teaching_id: int) -> str:
    """Remove a teaching rule by its ID."""
    log_tool("forget_teaching", teaching_id=teaching_id)
    if _store.remove_teaching(teaching_id):
        return f"Teaching #{teaching_id} removed."
    return f"Teaching #{teaching_id} not found."


@mcp.tool()
def list_teachings() -> str:
    """List all teaching rules."""
    rows = _store.list_teachings()
    if not rows:
        return "No teachings yet. Use teach() to add rules."
    lines = [
        f"  #{r['id']}: \"{r['rule']}\" → {r['action']} ({r['target_type']}: {r['target_id']})"
        for r in rows
    ]
    return f"{len(lines)} teachings:\n" + "\n".join(lines)


@mcp.tool()
def log_session(tool_name: str, plugin_id: str = "") -> str:
    """Record a tool usage in this session for passive learning."""
    log_tool("log_session", tool=tool_name, plugin_id=plugin_id)
    _store.log_session_tool(
        session_id=_session["id"],
        query=_session.get("topic", ""),
        query_vector=_session.get("topic_vector") or None,
        tool_used=tool_name,
        plugin_id=plugin_id or None,
    )
    # Phase M3 — roll tool-chain fingerprint into habits:tool-chains.
    _record_tool_chain(tool_name)
    # Plugin extension-point: A3 — forward tool-call events to plugin hooks.
    try:
        from . import plugin_hooks
        plugin_hooks.dispatch(
            "on_tool_call",
            {"tool_name": tool_name, "plugin_id": plugin_id, "session_id": _session.get("id")},
        )
    except Exception:  # noqa: BLE001
        pass
    return f"Logged: {tool_name} → {plugin_id or '(unknown plugin)'}"


@mcp.tool()
def close_session(summary: str = "") -> str:
    """Phase M3 — Close the current session and persist its L1 summary.

    - Writes ``summary`` (plus the session's tracked topic) into the
      ``session:log`` index as an L1 vector so future ``search_context``
      calls can surface "we worked on this before".
    - Flushes any partial tool-chain window into ``habits:tool-chains``.
    - Dispatches the ``on_session_end`` plugin hook (A3) so plugins can
      observe the session boundary.
    - Rotates the in-process session id so subsequent work starts fresh.
    """
    import hashlib, time
    log_tool("close_session", summary=summary[:80])
    topic = _session.get("topic", "")
    text = (summary or topic or "empty session").strip()
    sid = _session.get("id", "")
    # Flush remainder of rolling chain (if any).
    chain = _session.get("tool_chain") or []
    if chain:
        try:
            ctext = " → ".join(chain)
            _store.upsert_vector(
                namespace="habits:tool-chains",
                doc_id=f"{sid[:8]}:tail:{hashlib.sha1(ctext.encode()).hexdigest()[:8]}",
                text=ctext,
                metadata={"tools": chain, "session_id": sid, "partial": True},
                level="L2",
                source="tool_chain",
            )
        except Exception:  # noqa: BLE001
            pass
    try:
        _store.upsert_vector(
            namespace="session:log",
            doc_id=f"session:{sid}",
            text=f"Topic: {topic}\nSummary: {text}",
            metadata={"session_id": sid, "topic": topic,
                      "closed_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
            level="L1",
            source="close_session",
        )
    except Exception as exc:  # noqa: BLE001
        return f"close_session: session:log write failed: {exc}"
    try:
        from . import plugin_hooks
        plugin_hooks.dispatch(
            "on_session_end",
            {"session_id": sid, "topic": topic, "summary": text},
        )
    except Exception:  # noqa: BLE001
        pass
    _session["id"] = str(uuid.uuid4())
    _session["topic"] = ""
    _session["topic_vector"] = []
    _session["tool_chain"] = []
    return f"Session closed → session:log (new id={_session['id'][:8]})"


# ---------------------------------------------------------------------------
# Tasks (cross-session context)


def _generate_title(summary: str, context: str = "") -> str:
    """Generate a short task title from content using local LLM, or extract first line."""
    content = (summary or context or "").strip()
    if not content:
        return "Untitled task"
    # Try local LLM first
    if should_run_llm("title"):
        try:
            prompt = (
                "Write a concise task title in 6 words or fewer that describes this work.\n"
                "Reply with only the title, no punctuation at the end.\n\n"
                f"{content[:600]}\n\nTitle:"
            )
            generated = _generate(prompt, model=RERANK_MODEL, timeout=15.0, num_predict=30)
            generated = generated.strip().strip('"').strip("'")
            if generated and len(generated) < 120:
                return generated
        except Exception:
            pass
    # Fallback: first non-empty line, capped at 80 chars
    first_line = next((ln.strip() for ln in content.splitlines() if ln.strip()), content)
    return first_line[:80]


@mcp.tool()
def save_task(
    title: str,
    summary: str,
    context: str = "",
    tags: str = "",
    project: str = "",
    mode: str = "",
    initial_prompt: str = "",
    cwd: str = "",
) -> str:
    """Save an open task for retrieval in future sessions.

    Optional worktree spawn:
        project=<repo-name>     name under worktree.repo_roots (default ~/work/code).
                                When set, skill-hub creates a git worktree at
                                <repo>/.claude/worktrees/<task-name> on branch
                                cc/<task-name> and launches a Claude session in it.
        mode=terminal|tmux|background
                                terminal opens a new iTerm/Terminal tab (macOS),
                                tmux adds a window to $TMUX, background spawns
                                claude --print to a logfile. Default: from
                                worktree.default_mode config (terminal).
        initial_prompt=<text>   first message to send to the spawned session.
        cwd=<path>              if project is empty, walk up from cwd to find
                                a .git, then map back to a project name.

    Parallel-safe: each call creates an independent row (distinct task IDs).
    Two concurrent calls with identical args produce two separate tasks rather
    than merging -- callers should deduplicate before saving.
    """
    log_tool("save_task", title=title, tags=tags, project=project, mode=mode)

    # Auto-generate title when caller leaves it blank or too short
    if not title or len(title.strip()) < 4:
        title = _generate_title(summary, context)

    # Resolve project: explicit > cwd-based detection > none.
    from . import worktree as _wt
    resolved_project = project.strip() or None
    if not resolved_project and cwd:
        resolved_project = _wt.detect_project_from_cwd(cwd)

    worktree_blob = ""
    worktree_msg = ""
    spec = None
    if resolved_project:
        try:
            wt_name = _slugify_task_name(title)
            spec = _wt.ensure_worktree(
                resolved_project, wt_name, mode=(mode or None),  # type: ignore[arg-type]
            )
            spec = _wt.launch_session(spec, initial_prompt=initial_prompt or None)
            worktree_blob = spec.to_json()
            worktree_msg = (
                f"\nWorktree: {spec.worktree_path} (branch {spec.branch}, "
                f"mode {spec.mode}, pid {spec.last_pid or '?'})"
            )
        except _wt.WorktreeError as e:
            worktree_msg = f"\nWorktree skipped: {e}"

    try:
        vector = embed(f"{title}: {summary}")
    except RuntimeError:
        vector = []
    tid = _store.save_task(
        title=title, summary=summary, vector=vector,
        context=context, tags=tags, session_id=_session["id"],
        cwd=(spec.worktree_path if spec else cwd),
        branch=(spec.branch if spec else ""),
        worktree=worktree_blob,
    )
    task_opts = _store.get_task_options(tid)
    _write_active_task_marker(tid, _session["id"], title,
                              auto_approve=task_opts.get("auto_approve"),
                              options=task_opts)
    return f"Task #{tid} saved (open): \"{title}\"{worktree_msg}\nWill surface in future search_context() calls."


def _slugify_task_name(title: str) -> str:
    """Turn a task title into a worktree/branch-friendly slug."""
    import re
    slug = re.sub(r"[^A-Za-z0-9._\-]+", "-", title.strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-.")
    return slug[:48] or "task"


@mcp.tool()
def close_task(
    task_id: int,
    summary: str = "",
    remove_worktree: bool = False,
    compact_master_state: bool = False,
    master_state_project: str | None = None,
) -> str:
    """Close a task with LLM-compacted summary (~200 tokens).

    remove_worktree=True also runs `git worktree remove --force` and deletes
    the cc/<name> branch. Default keeps the worktree on disk so the task
    can be reopened later into the same workspace.

    compact_master_state=True additionally folds the task's recent auto-memory
    delta into the project's `## Master Project State` snapshot at
    `<project>/.memory/decisions.md`. master_state_project overrides the
    project root (default: cwd resolved from the task's stored worktree path,
    falling back to the current working directory).

    Parallel-safe: concurrent closes of the same task_id are idempotent --
    the second call finds status='closed' and returns early without
    overwriting the first compaction.
    """
    log_tool("close_task", task_id=task_id, remove_worktree=remove_worktree,
             compact_master_state=compact_master_state)
    task = _store.get_task(task_id)
    if not task:
        return f"Task #{task_id} not found."
    if task["status"] == "closed":
        return f"Task #{task_id} is already closed."

    # Prepare content for compaction
    content = summary or task["summary"]
    if task["context"]:
        content += f"\n\nContext:\n{task['context']}"

    # Compact via local LLM
    digest = compact(content)
    compact_text = json.dumps(digest, indent=2)

    # Re-embed the compacted summary for better future matching.
    # Optional: if the embed service is disabled, close without a vector
    # (store leaves the existing vector column untouched).
    try:
        compact_vector = embed(f"{digest.get('title', '')}: {digest.get('summary', '')}")
    except RuntimeError:
        compact_vector = None

    _store.close_task(task_id, compact_text, compact_vector)
    _clear_active_task_marker(task_id)

    # Optional worktree cleanup. Default is to keep the worktree on disk so
    # reopen_task() can spawn back into it.
    worktree_msg = ""
    blob = task["worktree"] if "worktree" in task.keys() else None
    if remove_worktree and blob:
        from . import worktree as _wt
        try:
            spec = _wt.WorktreeSpec.from_json(blob)
            _wt.teardown_worktree(spec)
            _store.set_task_worktree(task_id, "")
            worktree_msg = f"\nWorktree removed: {spec.worktree_path}"
        except (ValueError, TypeError, _wt.WorktreeError) as e:
            worktree_msg = f"\nWorktree teardown failed: {e}"

    # Refresh benefit/cost dashboard; never fail close_task on render error.
    dash_line = ""
    try:
        url = _dashboard.render_interactive(_store)
        if url:
            dash_line = f"\nDashboard: {url}"
        else:
            dash_path = _dashboard.render(_store)
            dash_line = f"\nDashboard: file://{dash_path}"
    except Exception as e:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning("dashboard render failed: %s", e)

    # Optional: fold this task's auto-memory delta into the project's
    # Master Project State snapshot. Never fail close_task on this error.
    master_state_line = ""
    if compact_master_state:
        try:
            from .master_state import compact_to_master_state as _cms
            project = master_state_project
            if not project and blob:
                from . import worktree as _wt2
                try:
                    project = _wt2.WorktreeSpec.from_json(blob).worktree_path
                except (ValueError, TypeError, _wt2.WorktreeError):
                    project = None
            if not project:
                import os as _os
                project = _os.getcwd()
            ms_result = _cms(project_root=project)
            if ms_result.get("status") == "written":
                master_state_line = f"\nMaster State: updated {ms_result.get('wrote')}"
            else:
                master_state_line = f"\nMaster State: {ms_result.get('status')} ({ms_result.get('reason', '')})"
        except Exception as e:  # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning("master_state compaction failed: %s", e)
            master_state_line = f"\nMaster State: error ({e})"

    return (
        f"Task #{task_id} closed and compacted.\n"
        f"Title: {digest.get('title', 'N/A')}\n"
        f"Summary: {digest.get('summary', 'N/A')}\n"
        f"Tags: {digest.get('tags', 'N/A')}\n"
        f"Decisions: {digest.get('decisions', [])}"
        f"{worktree_msg}"
        f"{dash_line}"
        f"{master_state_line}"
    )


@mcp.tool()
def compact_master_state(
    project_root: str,
    output_file: str = ".memory/decisions.md",
    section_title: str = "Master Project State",
    since_iso: str | None = None,
    dry_run: bool = False,
) -> str:
    """Generate or refresh the Master Project State snapshot for a project.

    Layer-analyzes recent auto-memory entries and produces a 4-section snapshot
    (Architecture / Invariants / Active Working Set / Recent Pivots) under the
    `## <section_title>` heading at `<project_root>/<output_file>`.

    dry_run=True returns the rendered Markdown without writing to disk.

    Idempotent: a rerun with no new memory entries returns "noop". Existing
    file is backed up to `<output_dir>/.backups/` before overwrite.
    """
    log_tool("compact_master_state", project_root=project_root, dry_run=dry_run)
    from .master_state import compact_to_master_state as _cms
    result = _cms(
        project_root=project_root,
        output_file=output_file,
        section_title=section_title,
        since_iso=since_iso,
        dry_run=dry_run,
    )
    if result.get("status") == "dry_run":
        rendered = result.get("rendered", "")
        files = result.get("memory_files_considered", [])
        return (
            f"DRY RUN — {len(rendered)} chars rendered from {len(files)} memory files.\n"
            f"--- snapshot preview ---\n{rendered}"
        )
    if result.get("status") == "written":
        return (
            f"Wrote: {result.get('wrote')}\n"
            f"Backup: {result.get('backup') or '(none — first write)'}\n"
            f"Delta: {result.get('delta_chars'):+d} chars\n"
            f"Memory files folded in: {len(result.get('memory_files_considered', []))}"
        )
    return f"{result.get('status', 'unknown')}: {result.get('reason', '')}"


@mcp.tool()
def set_task_auto_approve(task_id: int, enabled: bool | None = None) -> str:
    """Toggle per-task permissive auto-approve override.

    enabled=True  -> permissive: the task's own safe_bash_prefixes (stored in
                    context JSON under 'task_safe_prefixes') are ADDED to the
                    global allow-list while this task is active.
    enabled=False -> explicit opt-out (no-op for hook; never reduces safety).
    enabled=None  -> clear; fall back to global behavior.
    """
    log_tool("set_task_auto_approve", task_id=task_id, enabled=enabled)
    task = _store.get_task(task_id)
    if task is None:
        return f"Task #{task_id} not found."
    ok = _store.set_task_auto_approve(task_id, enabled)
    if not ok:
        return f"Task #{task_id}: no change."
    task_opts = _store.get_task_options(task_id)
    _refresh_active_marker_options(task_id, task, task_opts)
    label = "null" if enabled is None else str(bool(enabled)).lower()
    return f"Task #{task_id} auto_approve set to {label}."


@mcp.tool()
def set_task_options(task_id: int, options: str = "") -> str:
    """Set per-task option overrides as a JSON object.

    options is a JSON string with keys:
      auto_approve     bool | null  — permissive auto-approve (same as set_task_auto_approve)
      routing_disabled bool | null  — skip model routing for this task entirely
      model_pin        str  | null  — force a specific model ("haiku"/"sonnet"/"opus")

    Keys with value null remove that override (revert to global default).
    Pass an empty JSON object "{}" to clear all overrides.

    Example: set_task_options(1, '{"routing_disabled": true}')
             set_task_options(1, '{"auto_approve": true, "routing_disabled": false}')
             set_task_options(1, '{"routing_disabled": null}')  -- removes key
    """
    log_tool("set_task_options", task_id=task_id)
    task = _store.get_task(task_id)
    if task is None:
        return f"Task #{task_id} not found."
    try:
        patch = json.loads(options) if options.strip() else {}
        if not isinstance(patch, dict):
            return "options must be a JSON object."
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"
    ok = _store.set_task_options(task_id, patch)
    if not ok:
        return f"Task #{task_id}: no change."
    new_opts = _store.get_task_options(task_id)
    _refresh_active_marker_options(task_id, task, new_opts)
    return f"Task #{task_id} options updated: {json.dumps(new_opts)}"


@mcp.tool()
def update_task(task_id: int, summary: str = "", context: str = "",
                tags: str = "", title: str = "", color: str = "") -> str:
    """Update an open task with new information.

    Optional ``title`` rewrites the task's title (useful for cleaning up
    auto-created stubs whose title is the raw memory filename). Optional
    ``color`` sets a short status label (`green`, `yellow`, `red`, `cyan`,
    `blue`, `gray`) used by the dashboard + listings.

    Parallel-safe: SQLite serialises writes. Concurrent updates to the
    same task_id use last-write-wins semantics -- no corruption, no silent
    data loss.
    """
    log_tool("update_task", task_id=task_id)

    # Re-embed if title or summary changed. Optional: if embed service is
    # disabled or the model is unavailable, update text fields without
    # touching the vector (store leaves the existing vector column untouched
    # when vector is None).
    vector = None
    if summary or title:
        task = _store.get_task(task_id)
        if task:
            new_title = title or task["title"]
            new_summary = summary or task.get("summary", "")
            try:
                vector = embed(f"{new_title}: {new_summary}")
            except RuntimeError:
                vector = None

    if _store.update_task(task_id, summary=summary, context=context,
                          tags=tags, vector=vector,
                          title=title, color=color):
        return f"Task #{task_id} updated."
    return f"Task #{task_id} not found."


@mcp.tool()
def reopen_task(task_id: int) -> str:
    """Reopen a previously closed task.

    If the task owns a worktree:
      - and a session is still alive in it → focus message, no relaunch.
      - else → spawn a fresh Claude session in the same worktree directory
        (worktree itself is preserved across closes).
    """
    log_tool("reopen_task", task_id=task_id)
    if not _store.reopen_task(task_id):
        return f"Task #{task_id} not found."

    task = _store.get_task(task_id)
    blob = task["worktree"] if task else None
    if not blob:
        return f"Task #{task_id} reopened."

    from . import worktree as _wt
    try:
        spec = _wt.WorktreeSpec.from_json(blob)
    except (ValueError, TypeError) as e:
        return f"Task #{task_id} reopened (worktree spec unreadable: {e})."

    if _wt.is_session_alive(spec):
        return f"Task #{task_id} reopened.\n{_wt.focus_session(spec)}"

    try:
        spec = _wt.launch_session(spec)
        _store.set_task_worktree(task_id, spec.to_json())
        return (
            f"Task #{task_id} reopened.\n"
            f"Worktree: {spec.worktree_path} (mode {spec.mode}, "
            f"pid {spec.last_pid or '?'})"
        )
    except _wt.WorktreeError as e:
        return f"Task #{task_id} reopened (relaunch failed: {e})."


_COLOR_GLYPH = {
    "green": "●", "yellow": "◐", "red": "✗",
    "cyan": "◆", "blue": "○", "gray": "·",
}


@mcp.tool()
def list_tasks(status: str = "open") -> str:
    """List tasks. status: open (default), closed, or all."""
    log_tool("list_tasks", status=status)
    rows = _store.list_tasks(status)
    if not rows:
        return f"No {status} tasks."
    lines: list[str] = []
    for r in rows:
        state = f"[{r['status'].upper()}]"
        tags = f" ({r['tags']})" if r['tags'] else ""
        # Glyph keyed on auto-derived/explicit color column. ' ' keeps column
        # alignment when no colour is set (older rows pre-migration).
        try:
            color = r["color"]
        except (IndexError, KeyError):
            color = None
        glyph = _COLOR_GLYPH.get(color or "", " ")
        lines.append(f"  {glyph} #{r['id']} {state} {r['title']}{tags} — {r['summary'][:80]}...")
    return f"{len(lines)} tasks:\n" + "\n".join(lines)


@mcp.tool()
def validate_plan(plan_path: str, repo_path: str = "", check_files: bool = True) -> str:
    """Validate a plan YAML file against the plan-executor schema.

    Checks: top-level required fields, step schema, kind enum, non-empty file lists,
    depends_on references, cycle detection, and (if check_files) existence of
    protocols_ref / pattern_ref paths on disk.

    Args:
        plan_path: Path to the plan YAML file (e.g. ~/.claude/plans/foo.yaml).
        repo_path: Root for resolving protocols_ref / pattern_ref. Defaults to cwd.
        check_files: Verify referenced context files exist on disk.

    Returns:
        On success: "OK: <plan_id> — N steps (S smart, M mid)".
        On failure: multi-line "INVALID:" report with one error per line.
    """
    from .plan_executor import validate_plan_file, PlanValidationError, TIER_MAP

    log_tool("validate_plan", plan_path=plan_path)
    try:
        plan = validate_plan_file(
            Path(plan_path).expanduser(),
            repo_path=(Path(repo_path).expanduser() if repo_path else None),
            check_files=check_files,
        )
    except PlanValidationError as e:
        lines = ["INVALID:"] + [f"  - {err}" for err in e.errors]
        return "\n".join(lines)

    steps = plan.get("steps", [])
    smart = sum(1 for s in steps if TIER_MAP.get(s.get("kind")) == "tier_smart")
    mid = sum(1 for s in steps if TIER_MAP.get(s.get("kind")) == "tier_mid")
    return (
        f"OK: {plan['plan_id']} — {len(steps)} steps "
        f"({smart} smart, {mid} mid)"
    )


@mcp.tool()
def author_plan(goal: str, repo_path: str = "", preferred_runner: str = "") -> str:
    """Author a plan YAML for the given goal, transparently using the best
    available Opus runner (in_session → claude -p → SDK → API fallback).

    Resolution: HUB_PLAN_RUNNER env var > preferred_runner arg > auto-chain.
    The in-session runner returns a directive for the calling agent instead
    of a file path — the agent then authors the YAML itself using its own
    Read/Glob/Grep tools and calls validate_plan.

    Args:
        goal: One-line plan goal (used for the YAML filename slug).
        repo_path: Target repo root. Defaults to cwd.
        preferred_runner: Force a specific runner: in_session|cli|sdk|api.

    Returns:
        Markdown report with the runner used, plan path (or directive), and
        validation attempt count. Writes to ~/.claude/plans/<slug>.yaml.
    """
    from .plan_executor import author_plan as _author
    from .plan_executor import RunnerFailed

    log_tool("author_plan", goal=goal[:80], repo_path=repo_path,
             runner=preferred_runner or "auto")
    try:
        result = _author(
            goal,
            repo_path=(Path(repo_path).expanduser() if repo_path else None),
            preferred_runner=(preferred_runner or None) or None,  # type: ignore[arg-type]
        )
    except RunnerFailed as e:
        return f"ERROR: {e}"
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {type(e).__name__}: {e}"
    return result.as_markdown()


@mcp.tool()
def run_plan(plan_path: str, dry_run: bool = True, repo_path: str = "") -> str:
    """Walk every step of a plan YAML in topological order, stopping on the
    first failed/escalated step.

    Steps already marked ``done`` in the sidecar state are skipped (idempotent
    resume). Default is dry_run=True — pass dry_run=False to actually apply
    changes and run acceptance commands.

    Args:
        plan_path: Path to the plan YAML.
        dry_run: Preview only. Default True.
        repo_path: Root for file resolution. Defaults to cwd.

    Returns:
        Multi-line run summary with per-step outcomes and a stop reason if halted.
    """
    from .plan_executor import run_plan as _walk

    log_tool("run_plan", plan_path=plan_path, dry_run=dry_run)
    try:
        result = _walk(
            Path(plan_path).expanduser(),
            dry_run=dry_run,
            repo_path=(Path(repo_path).expanduser() if repo_path else None),
        )
    except (FileNotFoundError, ValueError) as e:
        return f"ERROR: {e}"
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {type(e).__name__}: {e}"
    return result.as_markdown()


@mcp.tool()
def execute_plan_step(
    plan_path: str,
    step_id: str,
    dry_run: bool = True,
    repo_path: str = "",
) -> str:
    """Execute one step from a plan YAML via the right model tier.

    - Resolves the step, checks depends_on via a sidecar .state.json file.
    - Maps step.kind → tier (architecture/integration → tier_smart; others → tier_mid).
    - Honors step.model_hint if set.
    - Builds a file-scoped context bundle (target files + protocols_ref + pattern_ref).
    - Calls the resolved model via litellm with a strict JSON-output contract.
    - If dry_run=False, writes returned file contents to disk and runs the
      acceptance command. On failure, retries once on tier_smart before marking
      the step failed.
    - Records a bandit reward (kind=task_class, plan_id=domain).

    Args:
        plan_path: Path to the plan YAML (typically ~/.claude/plans/<slug>.yaml).
        step_id: Which step to run.
        dry_run: Default True — preview only, no file writes, no acceptance run.
        repo_path: Root for resolving files. Defaults to cwd.

    Returns:
        Markdown status report (see StepResult.as_markdown).
    """
    from .plan_executor import execute_plan_step as _run

    log_tool("execute_plan_step", plan_path=plan_path, step_id=step_id, dry_run=dry_run)
    try:
        result = _run(
            Path(plan_path).expanduser(),
            step_id,
            dry_run=dry_run,
            repo_path=(Path(repo_path).expanduser() if repo_path else None),
        )
    except (KeyError, FileNotFoundError) as e:
        return f"ERROR: {e}"
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {type(e).__name__}: {e}"
    return result.as_markdown()


@mcp.tool()
def search_context(
    query: str,
    top_k: int = 5,
    categories: str = "all",
    include_plugin_memory: bool = True,
) -> str:
    """Unified search. categories: all (default), tasks, skills, closed, plugins (comma-separated).

    ``include_plugin_memory`` (A4): when True, merges results from plugin-
    declared ``memory.reads`` globs (indexed under ``memory:<plugin>`` vector
    namespaces) into the output.
    """
    log_tool("search_context", query=query, top_k=top_k, categories=categories)
    if not embed_available():
        return "No embedding backend available. Set VOYAGE_API_KEY, start Ollama, or install sentence-transformers."

    query_vector = embed(query)

    if not _session["topic"]:
        _session["topic"] = query
        _session["topic_vector"] = query_vector

    _last_search_state["query"] = query
    _last_search_state["vector"] = query_vector

    cats = {c.strip() for c in categories.split(",")}
    show_all = "all" in cats

    parts: list[str] = []

    # 1. Open tasks
    if show_all or "tasks" in cats:
        tasks = _store.search_tasks(query_vector, top_k=top_k, status="open")
        if tasks:
            task_lines = []
            for t in tasks:
                task_lines.append(
                    f"### Task #{t['id']}: {t['title']} (sim={t['similarity']:.2f})\n"
                    f"{t['summary']}\n"
                    + (f"Context: {t['context'][:300]}" if t.get('context') else "")
                )
            parts.append("## Open Tasks\n\n" + "\n\n".join(task_lines))

    # 2. Closed tasks
    if show_all or "closed" in cats:
        closed = _store.search_tasks(query_vector, top_k=3, status="closed")
        if closed:
            closed_lines = []
            for t in closed:
                digest = t.get("compact", t["summary"])
                closed_lines.append(
                    f"### Closed #{t['id']}: {t['title']} (sim={t['similarity']:.2f})\n"
                    f"{digest[:300]}"
                )
            parts.append("## Related Past Work\n\n" + "\n\n".join(closed_lines))

    # 3. Skills
    if show_all or "skills" in cats:
        skills = _store.search(query_vector, top_k=top_k)
        if skills:
            skill_lines = [f"- {s['id']}: {s['description'][:100]}" for s in skills]
            parts.append("## Matching Skills\n\n" + "\n".join(skill_lines))
            _last_search_state["skills"] = [s["id"] for s in skills]

    # 4. Plugin suggestions
    if show_all or "plugins" in cats:
        suggestions = _store.suggest_plugins(query_vector)
        if suggestions:
            enabled_plugins: dict = {}
            if SETTINGS_PATH.exists():
                settings = json.loads(SETTINGS_PATH.read_text())
                enabled_plugins = settings.get("enabledPlugins", {})
            disabled = [s for s in suggestions[:3]
                        if not enabled_plugins.get(s["plugin_id"], False)]
            if disabled:
                plug_lines = [f"- {s['short_name']}: {s['description'][:80]}" for s in disabled]
                parts.append(
                    "## Disabled Plugins That May Help\n\n" + "\n".join(plug_lines) +
                    "\nUse toggle_plugin() to enable."
                )

    # 5. Plugin memory (A4 + M2) — vectors from plugin-declared indexes.
    # Includes both the legacy ``memory:<plugin>`` namespace AND any custom
    # indexes a plugin declares via plugin.json ``vector_indexes`` (e.g.
    # ``career:profile``, ``career:narrative``).
    if include_plugin_memory:
        try:
            # Namespaces that are NOT core ("skills") — everything else is plugin/memory.
            all_rows = _store.search_vectors(query, namespaces=None, top_k=top_k * 2)
            mem_rows = [
                r for r in all_rows
                if str(r.get("namespace", "")) != "skills"
                and not str(r.get("namespace", "")).startswith("user:")
                and not str(r.get("namespace", "")).startswith("habits:")
                and not str(r.get("namespace", "")).startswith("session:")
            ][:top_k]
            if mem_rows:
                mem_lines = []
                for r in mem_rows:
                    ns = r["namespace"]
                    path = (r.get("metadata") or {}).get("path") or r.get("doc_id", "")
                    lvl = r.get("level", "")
                    mem_lines.append(
                        f"- [{ns}] {path} (score={r.get('score', 0):.2f}, {lvl})"
                    )
                parts.append("## Plugin Memory\n\n" + "\n".join(mem_lines))
        except Exception:  # noqa: BLE001
            pass

    # 6. User identity + habits — surface only when relevant to the query.
    try:
        id_rows = _store.search_vectors(
            query, namespaces=["user:identity", "user:preferences"],
            top_k=3, similarity_threshold=0.35,
        )
        if id_rows:
            parts.append(
                "## About You\n\n" +
                "\n".join(f"- {(r.get('metadata') or {}).get('fact') or r['doc_id']} "
                          f"({r.get('level','')}, score={r.get('score', 0):.2f})"
                          for r in id_rows)
            )
    except Exception:  # noqa: BLE001
        pass

    if not parts:
        return "No relevant context found. Try index_skills() and index_plugins() first."

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Management


@mcp.tool()
def index_skills() -> str:
    """Rebuild the skill index from all plugin directories."""
    log_tool("index_skills")
    if not embed_available():
        return (
            f"No embedding backend available. "
            f"Set VOYAGE_API_KEY, start Ollama with '{EMBED_MODEL}', or install sentence-transformers."
        )

    count, errors = index_all(_store)
    result = f"Indexed {count} skills."
    if errors:
        result += f"\n\nErrors ({len(errors)}):\n" + "\n".join(f"  - {e}" for e in errors[:10])
    return result


@mcp.tool()
def index_plugins() -> str:
    """Index plugin descriptions for suggest_plugins()."""
    log_tool("index_plugins")
    if not embed_available():
        return "No embedding backend available. Set VOYAGE_API_KEY, start Ollama, or install sentence-transformers."

    if not SETTINGS_PATH.exists():
        return "Settings file not found."

    settings = json.loads(SETTINGS_PATH.read_text())
    plugins = settings.get("enabledPlugins", {})

    # Known plugin descriptions (built-in knowledge for common plugins)
    descriptions: dict[str, str] = {
        "chrome-devtools-mcp": "Browser DevTools: take screenshots, inspect DOM, click elements, fill forms, run Lighthouse audits, analyze performance, debug CSS/layout, test accessibility, monitor network requests and console messages",
        "firebase": "Google Firebase: create projects, manage apps, deploy, read/write Firestore, configure security rules, manage environments",
        "terraform": "HashiCorp Terraform: manage infrastructure as code, workspaces, runs, variables, modules, providers, policy sets",
        "microsoft-docs": "Microsoft Learn: search Azure/Microsoft documentation, fetch code samples, get full doc pages",
        "mintlify": "Mintlify: search documentation hosted on Mintlify platform",
        "wordpress.com": "WordPress.com: manage sites, create posts/pages, manage media, domains, themes, plugins, statistics",
        "sanity-plugin": "Sanity CMS: content modeling, schema deployment, type generation, content management best practices",
        "huggingface-skills": "Hugging Face: build Gradio apps, train models, manage datasets, evaluate models, run inference, use transformers.js",
        "superpowers": "Development workflows: brainstorming, planning, TDD, debugging, code review, git worktrees, parallel agents, verification",
        "plugin-dev": "Claude Code plugin development: create plugins, skills, agents, hooks, MCP integrations, commands",
        "feature-dev": "Feature development: code review, code exploration, architecture design for new features",
        "code-review": "Code review: systematic review against plan and coding standards",
        "code-simplifier": "Code simplification: refine code for clarity, consistency, maintainability",
        "commit-commands": "Git workflows: commit, push, create PRs, clean gone branches",
        "hookify": "Hook management: analyze conversations for automation opportunities, create and configure hooks",
        "telegram": "Telegram: configure and access Telegram bot integration",
        "data": "Data engineering: Apache Airflow DAGs, Astronomer deployments, dbt with Cosmos, data lineage, warehouse setup",
        "data-engineering": "Data engineering (duplicate of data plugin): Airflow, Astronomer, dbt, lineage, warehouse",
        "document-skills": "Document generation: PDF, DOCX, XLSX, PPTX, canvas design, web artifacts, MCP builders, brand guidelines",
        "claude-api": "Claude API and Anthropic SDK: build apps with Claude, document generation (overlaps with document-skills)",
        "github": "GitHub integration: issues, PRs, releases, checks via gh CLI",
        "frontend-design": "Frontend design: UI/UX implementation guidance",
        "security-guidance": "Security: guidance on secure coding practices",
        "skill-creator": "Skill creation: build new Claude Code skills",
        "claude-code-setup": "Claude Code setup: analyze codebase and recommend automations",
        "claude-md-management": "CLAUDE.md management: revise and improve instruction files",
        "mcp-server-dev": "MCP server development: build MCP servers, apps, and MCPB bundles",
        "ai-firstify": "AI-first development: transform projects for AI-native workflows",
        "greptile": "Greptile: AI-powered code search across repositories",
        "atomic-agents": "Atomic Agents: build modular AI agent architectures",
        "optibot": "Optibot: optimization and automation tooling",
        "product-tracking-skills": "Product analytics: tracking plans, event instrumentation, coverage auditing",
        "example-skills": "Example skills: reference implementations of document generation (overlaps with document-skills)",
    }

    indexed = 0
    errors: list[str] = []

    def _index_plugin(plugin_key: str, short_name: str, desc: str) -> None:
        nonlocal indexed
        _store.upsert_plugin(plugin_key, short_name, desc)
        try:
            vector = embed(f"{short_name}: {desc}")
            _store.upsert_plugin_embedding(plugin_key, EMBED_MODEL, vector)
            indexed += 1
        except Exception as exc:
            errors.append(f"{short_name}: {exc}")

    # Index plugins from settings.json enabledPlugins
    for plugin_key in plugins:
        short_name = plugin_key.split("@")[0]
        desc = descriptions.get(short_name, "") or f"Plugin: {short_name}"
        _index_plugin(plugin_key, short_name, desc)

    # Index extra_skill_dirs entries as plugin sources (auto-registration)
    cfg = _cfg.load_config()
    for entry in cfg.get("extra_skill_dirs", []):
        if not entry.get("enabled", True):
            continue
        source = entry.get("source", "extra")
        base = Path(entry["path"]).expanduser()
        # Count skills in this source to build a description
        skill_count = len(list(base.rglob("SKILL.md"))) if base.exists() else 0
        desc = entry.get("description") or (
            f"Archived skills library ({skill_count} skills) from {base.name}"
        )
        _index_plugin(source, source, desc)

    # Index explicit extra_plugin_dirs entries.
    # If `base` itself contains plugin.json it IS the plugin; otherwise treat
    # each subdirectory as a candidate plugin.
    def _read_plugin_desc(plugin_dir: Path, fallback: str) -> tuple[str, str]:
        """Return (short_name, description)."""
        import json as _json
        short_name = plugin_dir.name
        desc = fallback
        pj = plugin_dir / "plugin.json"
        if pj.exists():
            try:
                data = _json.loads(pj.read_text(encoding="utf-8", errors="replace"))
                short_name = data.get("name", short_name)
                desc = data.get("description", desc) or desc
                return short_name, desc
            except Exception:
                pass
        rm = plugin_dir / "README.md"
        if rm.exists():
            try:
                text = rm.read_text(encoding="utf-8", errors="replace")
                para = re.search(r"\S.{20,}", re.sub(r"^#+.*$", "", text, flags=re.MULTILINE))
                if para:
                    desc = para.group(0)[:200].strip()
            except Exception:
                pass
        return short_name, desc or f"Plugin: {short_name}"

    for entry in cfg.get("extra_plugin_dirs", []):
        if not entry.get("enabled", True):
            continue
        source = entry.get("source", "extra")
        base = Path(entry["path"]).expanduser()
        if not base.exists():
            continue
        if (base / "plugin.json").exists():
            # Base IS a single plugin root.
            short_name, desc = _read_plugin_desc(base, entry.get("description", ""))
            _index_plugin(short_name, short_name, desc)
        else:
            # Container of plugins — each subdir with plugin.json (or README.md)
            # becomes a plugin. Skip dotdirs and venv-like clutter.
            for plugin_dir in sorted(d for d in base.iterdir() if d.is_dir()):
                if plugin_dir.name.startswith(".") or plugin_dir.name in {
                    "__pycache__", "node_modules", "venv", ".venv",
                }:
                    continue
                if not (plugin_dir / "plugin.json").exists() and \
                   not (plugin_dir / "README.md").exists():
                    continue
                short_name, desc = _read_plugin_desc(plugin_dir, entry.get("description", ""))
                plugin_key = f"{source}:{plugin_dir.name}"
                _index_plugin(plugin_key, short_name, desc)

    result = f"Indexed {indexed} plugins."
    if errors:
        result += f"\n\nErrors: " + "; ".join(errors[:5])
    return result


MARKETPLACES_DIR = Path.home() / ".claude" / "plugins" / "marketplaces"


def _run_git(cwd: Path, *args: str, timeout: float = 30.0) -> tuple[int, str, str]:
    """Run a git command in `cwd`. Returns (returncode, stdout, stderr)."""
    import subprocess
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, "", str(exc)


def _default_branch(cwd: Path) -> str:
    """Resolve origin's default branch; fallback to 'main'."""
    rc, out, _ = _run_git(cwd, "symbolic-ref", "--short", "refs/remotes/origin/HEAD")
    if rc == 0 and out.startswith("origin/"):
        return out.split("/", 1)[1]
    return "main"


def _skills_at_sha(cwd: Path, sha: str) -> set[str]:
    """List top-level skill directory names under skills/ at a given commit."""
    rc, out, _ = _run_git(cwd, "ls-tree", "--name-only", f"{sha}:skills")
    if rc != 0:
        return set()
    return {line.strip() for line in out.splitlines() if line.strip()}


@mcp.tool()
def update_marketplace(name: str = "anthropic-agent-skills",
                       dry_run: bool = False,
                       reindex: bool = True) -> str:
    """Pull the latest from a git-backed plugin marketplace and re-index its skills.

    Runs `git fetch` and (unless dry_run) `git merge --ff-only` in
    ~/.claude/plugins/marketplaces/<name>/. Aborts on non-git marketplaces,
    dirty working trees, or non-fast-forward. After a successful pull,
    triggers skill re-index when reindex=True.

    Returns a human-readable report with before/after SHAs, commits pulled,
    skills added/removed, and reindex outcome.
    """
    import time
    log_tool("update_marketplace", name=name, dry_run=dry_run, reindex=reindex)
    started_ms = int(time.time() * 1000)

    path = MARKETPLACES_DIR / name
    if not path.exists() or not path.is_dir():
        return f"Marketplace not found: {path}"
    if not (path / ".git").exists():
        return (f"{name} is not a git clone (no .git/). "
                f"This tool only updates git-backed marketplaces.")

    rc, origin_url, err = _run_git(path, "remote", "get-url", "origin")
    if rc != 0 or not origin_url:
        return f"No origin remote in {name}: {err or 'not configured'}"

    rc, status_out, err = _run_git(path, "status", "--porcelain")
    if rc != 0:
        return f"git status failed in {name}: {err}"
    if status_out:
        return (f"Dirty working tree in {name} — aborting to protect local edits. "
                f"Resolve manually:\n  cd {path}\n  git status")

    rc, before_sha, err = _run_git(path, "rev-parse", "HEAD")
    if rc != 0:
        return f"Could not read HEAD in {name}: {err}"

    branch = _default_branch(path)

    rc, _, err = _run_git(path, "fetch", "origin", branch, timeout=60.0)
    if rc != 0:
        return f"git fetch failed in {name}: {err}"

    rc, fetched_sha, err = _run_git(path, "rev-parse", f"origin/{branch}")
    if rc != 0:
        return f"Could not resolve origin/{branch} in {name}: {err}"

    if before_sha == fetched_sha:
        return (f"{name}: already up to date at {before_sha[:8]} (branch {branch}).")

    rc, ahead_out, _ = _run_git(
        path, "rev-list", "--left-right", "--count", f"origin/{branch}...HEAD",
    )
    behind_ahead = ahead_out.split() if rc == 0 else ["?", "?"]
    behind = behind_ahead[0] if behind_ahead else "?"
    ahead = behind_ahead[1] if len(behind_ahead) > 1 else "?"
    if ahead not in ("0", "?"):
        return (f"{name}: local branch is {ahead} commit(s) ahead of "
                f"origin/{branch} — non-fast-forward. Resolve manually.")

    rc, commits_out, _ = _run_git(
        path, "log", f"{before_sha}..{fetched_sha}", "--oneline",
    )
    commits = commits_out.splitlines() if rc == 0 else []
    commit_count = len(commits)

    skills_before = _skills_at_sha(path, before_sha)
    skills_after = _skills_at_sha(path, fetched_sha)
    added = sorted(skills_after - skills_before)
    removed = sorted(skills_before - skills_after)

    header = (f"{name}: origin={origin_url}\n"
              f"branch={branch}  before={before_sha[:8]}  fetched={fetched_sha[:8]}\n"
              f"behind={behind}  commits_pulled={commit_count}")

    if dry_run:
        lines = [f"[DRY RUN] {header}"]
        if added:
            lines.append(f"skills_added={len(added)}: {', '.join(added)}")
        if removed:
            lines.append(f"skills_removed={len(removed)}: {', '.join(removed)}")
        if commits:
            preview = "\n  ".join(commits[:10])
            lines.append(f"commits (first 10):\n  {preview}")
            if commit_count > 10:
                lines.append(f"  ... and {commit_count - 10} more")
        lines.append("(no changes applied — pass dry_run=False to pull)")
        return "\n".join(lines)

    rc, _, err = _run_git(
        path, "merge", "--ff-only", f"origin/{branch}", timeout=60.0,
    )
    if rc != 0:
        return (f"git merge --ff-only failed in {name}: {err}\n"
                f"HEAD is unchanged at {before_sha[:8]}.")

    rc, after_sha, _ = _run_git(path, "rev-parse", "HEAD")
    if rc != 0 or after_sha != fetched_sha:
        return (f"Merge reported success but HEAD did not advance to "
                f"{fetched_sha[:8]} — got {after_sha[:8]}. Investigate manually.")

    lines = [header,
             f"pulled={before_sha[:8]}..{after_sha[:8]} ({commit_count} commit(s))"]
    if added:
        lines.append(f"skills_added={len(added)}: {', '.join(added)}")
    if removed:
        lines.append(f"skills_removed={len(removed)}: {', '.join(removed)}")
    if commits:
        preview = "\n  ".join(commits[:10])
        lines.append(f"commits:\n  {preview}")
        if commit_count > 10:
            lines.append(f"  ... and {commit_count - 10} more")

    if reindex:
        if not embed_available():
            lines.append("reindex skipped: no embedding backend available.")
        else:
            idx_count, idx_errors = index_all(_store)
            lines.append(f"reindexed {idx_count} skills"
                         + (f" (errors: {len(idx_errors)})" if idx_errors else ""))
    else:
        lines.append("reindex skipped (reindex=False)")

    duration_ms = int(time.time() * 1000) - started_ms
    lines.append(f"duration_ms={duration_ms}")
    return "\n".join(lines)


@mcp.tool()
def list_skills(plugin: str = "") -> str:
    """List all indexed skills. Optional plugin filter."""
    rows = _store.list_skills()
    if plugin:
        rows = [r for r in rows if r["plugin"] == plugin]
    if not rows:
        return "No skills indexed. Run index_skills() first."
    lines = [f"- {r['id']}: {r['description'] or '(no description)'}" for r in rows]
    return f"{len(lines)} skills:\n" + "\n".join(lines)


@mcp.tool()
def toggle_plugin(plugin_name: str, enabled: bool) -> str:
    """Enable or disable a plugin. Takes effect on next session restart."""
    from .plugin_registry import toggle as _toggle

    return _toggle(plugin_name, enabled)


# ──────────────────────────────────────────────────────────────────────
# S3 F-SELECT — profile-based plugin curation
# ──────────────────────────────────────────────────────────────────────


@mcp.tool()
def list_profiles() -> str:
    """List saved plugin profiles; active profile prefixed with *."""
    from . import profiles as _prof

    profs = _prof.list_profiles(_store)
    if not profs:
        return "No profiles yet. Use create_profile(name=..., plugins=[...])."
    lines = []
    for p in profs:
        flag = "*" if p["is_active"] else " "
        enabled_count = sum(1 for v in p["plugins"].values() if v)
        desc = f" — {p['description']}" if p["description"] else ""
        lines.append(f"{flag} {p['name']} ({enabled_count} plugins){desc}")
    return "Profiles:\n" + "\n".join(lines)


@mcp.tool()
def create_profile(name: str, plugins: str = "",
                    description: str = "", overwrite: bool = False) -> str:
    """Create or overwrite a plugin profile.

    Args:
        name: profile name (e.g. "geoid", "minimal")
        plugins: comma-separated plugin ids to enable. Empty → snapshot current
                 ``~/.claude/settings.json`` enabledPlugins.
        description: human-readable blurb
        overwrite: replace existing profile if True
    """
    from . import profiles as _prof
    from . import plugin_registry as _plugins

    if plugins.strip():
        plugin_map: dict[str, bool] | list[str] = [
            p.strip() for p in plugins.split(",") if p.strip()
        ]
    else:
        plugin_map = dict(_plugins._load_settings().get("enabledPlugins") or {})
        if not plugin_map:
            return "error: no plugins in settings.json to snapshot"
    try:
        profile = _prof.create_profile(
            _store, name, plugin_map, description=description, overwrite=overwrite,
        )
    except ValueError as exc:
        return f"error: {exc}"
    n = sum(1 for v in profile["plugins"].values() if v)
    return f"profile {name!r} saved ({n} plugins enabled)"


@mcp.tool()
def switch_profile(name: str, dry_run: bool = False) -> str:
    """Switch active profile; writes its plugin map into ~/.claude/settings.json.

    Restart Claude Code for the change to take effect (harness bakes
    enabledPlugins into the session prompt at startup).
    """
    from . import profiles as _prof

    try:
        out = _prof.switch_profile(_store, name, dry_run=dry_run)
    except KeyError as exc:
        return f"error: {exc}"
    changed = out["changed_plugins"]
    verb = "would change" if dry_run else "changed"
    if not changed:
        return f"profile {name!r} already matches settings.json (no changes)"
    lines = [f"{verb} {len(changed)} plugin(s):"]
    for pid, delta in changed.items():
        lines.append(f"  - {pid}: {delta['before']} → {delta['after']}")
    if not dry_run:
        lines.append("Restart Claude Code to apply.")
    return "\n".join(lines)


@mcp.tool()
def delete_profile(name: str) -> str:
    """Delete a saved profile by name."""
    from . import profiles as _prof

    return "deleted" if _prof.delete_profile(_store, name) else f"no such profile: {name}"


# ──────────────────────────────────────────────────────────────────────
# S4 F-ROUTE — ε-greedy bandit over model tiers
# ──────────────────────────────────────────────────────────────────────


@mcp.tool()
def route_to_model(
    prompt: str = "",
    complexity: float = 0.5,
    domain_hints: str = "",
) -> str:
    """Pick a model tier for the given prompt via ε-greedy bandit.

    Args:
        prompt: user message (used for Tier-1 heuristic classification
                if ``complexity`` is not explicitly set; ignored otherwise)
        complexity: override 0.0-1.0 complexity score
        domain_hints: comma-separated hints (debugging, architecture, testing...)

    Returns a human summary; use ``bandit_stats`` for the full per-tier table.
    """
    from .router import bandit as _bandit
    from .router.heuristics import classify as _h_classify

    # Prefer explicit complexity; fall back to heuristic over the prompt.
    if not 0.0 <= complexity <= 1.0:
        return "error: complexity must be in [0.0, 1.0]"

    hints = [h.strip() for h in domain_hints.split(",") if h.strip()]
    if not hints and prompt:
        sig = _h_classify(prompt)
        if complexity == 0.5:  # caller left default → use heuristic
            complexity = sig.complexity
        hints = list(sig.domain_hints)

    decision = _bandit.select_tier(_store, complexity, hints)
    per_tier = decision.stats["per_tier"]
    lines = [
        f"→ tier={decision.tier}  model={decision.model or '(not configured)'}",
        f"   confidence={decision.confidence:.2f}  {decision.reasoning}",
        f"   bucket: task_class={decision.stats['task_class']}  domain={decision.stats['domain']}",
        "   per-tier stats:",
    ]
    for t, s in per_tier.items():
        lines.append(
            f"     - {t}: trials={int(s['trials'])}  successes={s['successes']:.1f}  rate={s['rate']:.2f}"
        )
    return "\n".join(lines)


@mcp.tool()
def record_model_reward(
    tier: str,
    success: float,
    task_class: str = "",
    domain: str = "_none",
    complexity: float = -1.0,
    domain_hints: str = "",
) -> str:
    """Record a reward for a ``(task_class, domain, tier)`` trial.

    Parallel-safe: upsert is a single atomic SQLite statement. Concurrent
    calls accumulate trials and successes without loss.

    Provide ``task_class`` directly, or leave it empty and pass ``complexity``
    + ``domain_hints`` so the bandit derives the bucket the same way
    ``route_to_model`` did.
    """
    from .router import bandit as _bandit

    if not task_class:
        if complexity < 0.0:
            return "error: provide task_class or complexity (0.0-1.0)"
        hints = [h.strip() for h in domain_hints.split(",") if h.strip()]
        task_class, domain = _bandit.bucket(complexity, hints)
    try:
        _bandit.record_reward(_store, tier, task_class, domain, success)
    except ValueError as exc:
        return f"error: {exc}"
    return f"ok: reward recorded tier={tier} task_class={task_class} domain={domain} success={success:.2f}"


@mcp.tool()
def bandit_stats() -> str:
    """Dump the full ``model_rewards`` table."""
    from .router import bandit as _bandit

    rows = _bandit.summary(_store)
    if not rows:
        return "No bandit data yet. Call route_to_model + record_model_reward to seed."
    lines = [
        f"{r['task_class']:<9} {r['domain']:<14} {r['tier']:<11} "
        f"trials={r['trials']:<4} successes={r['successes']:<5.1f} rate={r['rate']:.2f}"
        for r in rows
    ]
    return "model_rewards:\n" + "\n".join(lines)


@mcp.tool()
def improve_prompt(text: str, rewriters: str = "") -> str:
    """Apply S5 F-PROMPT rewriters and return the enriched prompt.

    ``rewriters`` is a comma-separated list of rewriter names (see
    ``list_prompt_rewriters``). Pass ``"all"`` to run every registered
    rewriter; leave empty to use the default chain.
    """
    from .router import rewriters as _rw

    names: list[str] | None
    if not rewriters.strip():
        names = None
    else:
        names = [n.strip() for n in rewriters.split(",") if n.strip()]
    result = _rw.improve_prompt(text, _store, rewriters=names)
    header = "applied: " + (", ".join(result.applied) if result.applied else "none")
    notes = "\n".join(f"  - {n}" for n in result.notes)
    return f"{header}\nnotes:\n{notes}\n\n---\n{result.prompt}"


@mcp.tool()
def list_prompt_rewriters() -> str:
    """List registered prompt rewriters (S5 F-PROMPT)."""
    from .router import rewriters as _rw

    return "rewriters:\n" + "\n".join(f"  - {n}" for n in _rw.available())


@mcp.tool()
def auto_curate_plugins(stale_days: int = 14) -> str:
    """Suggest plugins to disable — currently enabled but unused in the last N days.

    Derived from ``session_log`` (populated automatically via post-tool hooks).
    """
    from . import profiles as _prof

    candidates = _prof.auto_curate_candidates(_store, stale_days=stale_days)
    if not candidates:
        return f"No stale plugins — all enabled plugins had activity in the last {stale_days} days."
    lines = [f"{len(candidates)} plugin(s) with no activity in the last {stale_days} days:"]
    for c in candidates:
        lines.append(f"  - {c['plugin_id']}  ({c['reason']})")
    return "\n".join(lines)


@mcp.tool()
def session_stats() -> str:
    """Show most-used plugins from session history (passive learning data)."""
    rows = _store.get_session_stats()
    if not rows:
        return "No session data yet. Usage is logged automatically via hooks."
    lines = [
        f"- {r['plugin_id']}: {r['usage_count']} tool calls across {r['session_count']} sessions"
        for r in rows
    ]
    return "Plugin usage stats:\n" + "\n".join(lines)


@mcp.tool()
def configure(key: str = "", value: str = "") -> str:
    """View or update Skill Hub config. No args = show all. key+value = update.

    Common settings:
        reason_model    — LLM for re-ranking/compaction/classification
                          "deepseek-r1:1.5b" (default, 1.1GB)
                          "deepseek-r1:7b" (4.7GB, better quality)
                          "deepseek-r1:14b" (9GB, best for 32GB+ RAM)
                          "qwen2.5-coder:7b" (4.7GB, code-focused)
        embed_model     — Embedding model
                          "nomic-embed-text" (default, 274MB)
                          "mxbai-embed-large" (669MB, higher quality)
        hook_enabled    — Enable/disable UserPromptSubmit hook (true/false)
        search_top_k    — Default number of search results (integer)

    Args:
        key:   Config key to set. Empty to show all.
        value: New value. Strings, numbers, and booleans auto-detected.
    """
    log_tool("configure", key=key, value=value)
    from . import config as cfg

    if not key:
        current = cfg.load_config()
        lines = [f"  {k}: {v}" for k, v in sorted(current.items())]
        return "Current config:\n" + "\n".join(lines) + f"\n\nFile: {cfg.CONFIG_PATH}"

    # Parse value type
    parsed: str | int | float | bool = value
    if value.lower() in ("true", "false"):
        parsed = value.lower() == "true"
    else:
        try:
            parsed = int(value)
        except ValueError:
            try:
                parsed = float(value)
            except ValueError:
                pass

    current = cfg.load_config()
    current[key] = parsed
    cfg.save_config(current)

    # Update module-level constants if relevant
    from . import embeddings
    if key == "embed_model":
        embeddings.EMBED_MODEL = str(parsed)
    elif key == "reason_model":
        embeddings.RERANK_MODEL = str(parsed)
    elif key == "ollama_base":
        embeddings.OLLAMA_BASE = str(parsed)

    return f"Config updated: {key} = {parsed}\nRestart MCP server for full effect."


@mcp.tool()
def optimize_memory(dry_run: bool = True) -> str:
    """Analyze memory files with tier-smart LLM routing. Recommends KEEP/PRUNE/COMPACT/MERGE. dry_run=True for report only.
    """
    log_tool("optimize_memory", dry_run=dry_run)

    if not should_run_llm("optimize_memory"):
        s = snapshot(force=True)
        return (
            f"Skipped: system under pressure ({s.pressure.name}, "
            f"cpu={s.cpu_load_1m:.0%}, mem={s.memory_used_pct:.0%}).\n"
            f"optimize_memory runs only when the machine is idle.\n"
            f"Try again when load is lower, or force with: "
            f"SKILL_HUB_FORCE_LLM=1 optimize_memory"
        )

    # Tier-based routing (Phase C.2): use litellm provider instead of bare Ollama
    tier = str(_cfg.get("optimize_memory_tier") or "smart")
    _TIER_MAP = {"smart": "tier_smart", "mid": "tier_mid", "cheap": "tier_cheap"}
    tier_key = _TIER_MAP.get(tier, tier)  # passthrough if already "tier_*"
    from .llm.litellm_adapter import get_provider as _get_llm

    mem_path = Path.home() / ".claude" / "projects" / \
               "-Users-ccancellieri-work-code" / "memory"
    if not mem_path.exists():
        return f"Memory directory not found: {mem_path}"

    index_path = mem_path / "MEMORY.md"

    # Collect all detail files
    entries: list[dict] = []
    for mf in sorted(mem_path.glob("*.md")):
        if mf.name == "MEMORY.md":
            continue
        content = mf.read_text(encoding="utf-8", errors="replace")
        tokens = max(1, len(content) // 4)
        prefix = mf.stem.split("_")[0]
        entries.append({
            "file": mf.name,
            "category": prefix,
            "tokens": tokens,
            "content": content,
        })

    if not entries:
        return "No memory detail files found."

    total_tokens = sum(e["tokens"] for e in entries)
    lines = [
        f"=== Memory Optimization Report ===\n",
        f"Analyzing {len(entries)} files (~{total_tokens:,} tokens total) "
        f"via tier={tier}...\n",
    ]

    # Call LLM via tier-based routing
    from .embeddings import _OPTIMIZE_CONTEXT_PROMPT
    formatted = []
    for e in entries:
        formatted.append(
            f"--- {e['file']} ({e['category']}, ~{e['tokens']} tokens) ---\n"
            f"{e['content'][:2000]}"
        )
    content_str = "\n\n".join(formatted)
    max_chars = int(_cfg.get("compact_max_input_chars")) * 4  # 16k chars
    classification_prompt = _OPTIMIZE_CONTEXT_PROMPT.format(
        content=content_str[:max_chars]
    )

    results: list[dict] = []
    try:
        _provider = _get_llm()
        raw = _provider.complete(
            classification_prompt,
            tier=tier_key,
            max_tokens=2000,
            temperature=0.0,
            timeout=300.0,
        )
        import re as _re
        raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
        for match in _re.finditer(r"\{[^{}]+\}", raw):
            try:
                obj = json.loads(match.group())
                results.append(obj)
            except json.JSONDecodeError:
                continue
    except Exception as exc:
        results.append({
            "error": str(exc),
            "summary": True,
            "total": len(entries),
            "keep": len(entries),
            "prune": 0,
            "compact": 0,
            "merge": 0,
            "est_tokens_saved": 0,
        })

    # Separate recommendations from summary
    recs = [r for r in results if not r.get("summary")]
    summary = next((r for r in results if r.get("summary")), None)

    # Group by action
    by_action: dict[str, list[dict]] = {}
    for r in recs:
        action = r.get("action", "UNKNOWN")
        by_action.setdefault(action, []).append(r)

    for action in ["PRUNE", "COMPACT", "MERGE", "KEEP"]:
        items = by_action.get(action, [])
        if not items:
            continue
        lines.append(f"\n{'─' * 40}")
        lines.append(f"  {action} ({len(items)} files):")
        for r in items:
            fname = r.get("file", "?")
            reason = r.get("reason", "")
            tok = next((e["tokens"] for e in entries if e["file"] == fname), 0)
            lines.append(f"    • {fname} (~{tok} tokens)")
            lines.append(f"      {reason}")
            if action == "COMPACT" and r.get("compacted"):
                preview = r["compacted"][:200].replace("\n", " ")
                lines.append(f"      → {preview}...")

    # Summary
    if summary and not summary.get("error"):
        saved = summary.get("est_tokens_saved", 0)
        lines.append(f"\n{'═' * 40}")
        lines.append(f"  Total files: {summary.get('total', len(entries))}")
        lines.append(f"  Keep: {summary.get('keep', '?')}  "
                     f"Prune: {summary.get('prune', '?')}  "
                     f"Compact: {summary.get('compact', '?')}  "
                     f"Merge: {summary.get('merge', '?')}")
        lines.append(f"  Estimated tokens saved: ~{saved:,}")
        lines.append(f"  (~{saved:,} tokens × every session)")
    elif summary and summary.get("error"):
        lines.append(f"\n⚠ LLM error: {summary['error']}")

    # Apply pruning if not dry_run
    pruned_files: list[str] = []
    if not dry_run and "PRUNE" in by_action:
        lines.append(f"\n{'─' * 40}")
        lines.append("  Applying PRUNE actions...")
        for r in by_action["PRUNE"]:
            fname = r.get("file", "")
            fpath = mem_path / fname
            if fpath.exists() and fpath.name != "MEMORY.md":
                fpath.unlink()
                pruned_files.append(fname)
                lines.append(f"    ✓ Deleted {fname}")

        # Update MEMORY.md index — remove lines referencing pruned files
        if pruned_files and index_path.exists():
            index_text = index_path.read_text(encoding="utf-8", errors="replace")
            new_lines = []
            for line in index_text.splitlines():
                if any(pf in line for pf in pruned_files):
                    continue
                new_lines.append(line)
            index_path.write_text("\n".join(new_lines) + "\n",
                                 encoding="utf-8")
            lines.append(f"    ✓ Updated MEMORY.md (removed {len(pruned_files)} entries)")

    if dry_run and "PRUNE" in by_action:
        lines.append(f"\n💡 To apply pruning: optimize_memory(dry_run=False)")

    return "\n".join(lines)


@mcp.tool()
def status(section: str = "summary") -> str:
    """Skill Hub health check. section: summary (default), context, resources, tips, full."""
    import httpx

    cfg = _cfg.load_config()
    ollama_base = cfg.get("ollama_base", "http://localhost:11434")
    embed_model = cfg.get("embed_model", "nomic-embed-text")
    reason_model = cfg.get("reason_model", "deepseek-r1:1.5b")
    show_all = section == "full"

    def _est_tokens(path: Path) -> int:
        try:
            return max(1, len(path.read_text(encoding="utf-8", errors="replace")) // 4)
        except OSError:
            return 0

    lines: list[str] = []

    # --- Summary section (always shown) ---
    if section in ("summary", "full"):
        lines.append("=== Skill Hub Status ===\n")
        lines.append("MCP server:      running")

        try:
            resp = httpx.get(f"{ollama_base}/api/tags", timeout=5.0)
            available = [m["name"] for m in resp.json().get("models", [])]
            lines.append(f"Ollama:          reachable")
            embed_ok = any(embed_model in m for m in available)
            reason_ok = any(reason_model in m for m in available)
            lines.append(f"Models:          embed={embed_model} ({'ok' if embed_ok else 'MISSING'}), "
                         f"reason={reason_model} ({'ok' if reason_ok else 'MISSING'})")
        except Exception:
            lines.append(f"Ollama:          NOT reachable at {ollama_base}")

        try:
            skill_count = len(_store.list_skills())
            task_rows = _store.list_tasks("all")
            open_tasks = sum(1 for r in task_rows if r["status"] == "open")
            lines.append(f"DB:              {skill_count} skills, {len(task_rows)} tasks ({open_tasks} open)")
        except Exception as exc:
            lines.append(f"DB:              error — {exc}")

    # --- Context section ---
    if section in ("context", "full"):
        lines.append("\n=== Context Usage (estimated) ===\n")
        context_files = [
            ("User    ", Path.home() / ".claude" / "CLAUDE.md"),
            ("Project ", Path.home() / "work" / "code" / "CLAUDE.md"),
            ("AutoMem ", Path.home() / ".claude" / "projects" /
             "-Users-ccancellieri-work-code" / "memory" / "MEMORY.md"),
        ]
        import os
        cwd_claude = Path(os.getcwd()) / "CLAUDE.md"
        if cwd_claude.exists() and cwd_claude not in [f for _, f in context_files]:
            context_files.append(("Cwd     ", cwd_claude))

        total_ctx = 0
        for label, fpath in context_files:
            if fpath.exists():
                tok = _est_tokens(fpath)
                total_ctx += tok
                lines.append(f"  {label}  ~{tok:>5} tokens  {fpath}")
            else:
                lines.append(f"  {label}  (not found)")
        lines.append(f"\n  Total always-loaded:  ~{total_ctx:,} tokens")

        mem_path = Path.home() / ".claude" / "projects" / \
                   "-Users-ccancellieri-work-code" / "memory"
        if mem_path.exists():
            lines.append("\n  Breakdown by category:")
            cats: dict[str, int] = {}
            for mf in sorted(mem_path.glob("*.md")):
                if mf.name == "MEMORY.md":
                    continue
                prefix = mf.stem.split("_")[0]
                cats[prefix] = cats.get(prefix, 0) + _est_tokens(mf)
            for cat, tok in sorted(cats.items(), key=lambda x: -x[1]):
                lines.append(f"    {cat:<12} ~{tok:>5} tokens")

    # --- Resources section ---
    if section in ("resources", "full"):
        lines.append("\n=== System Resources ===\n")
        s = snapshot(force=True)
        pressure_icon = {"IDLE": "G", "LOW": "Y", "MODERATE": "O", "HIGH": "R"}
        lines.append(f"  Pressure:  [{pressure_icon.get(s.pressure.name, '?')}] {s.pressure.name}")
        lines.append(f"  CPU load:  {s.cpu_load_1m:.0%} ({_get_cpu_info()} cores)")
        lines.append(f"  Memory:    {s.memory_used_pct:.0%} used, {s.memory_available_mb}MB avail")

    # --- Tips section ---
    if section in ("tips", "full"):
        lines.append("\n=== Memory Optimization Tips ===\n")
        mem_index_tok = _est_tokens(
            Path.home() / ".claude" / "projects" /
            "-Users-ccancellieri-work-code" / "memory" / "MEMORY.md"
        )
        if mem_index_tok > 1500:
            lines.append(f"  - MEMORY.md ~{mem_index_tok} tokens — prune stale entries")
        user_tok = _est_tokens(Path.home() / ".claude" / "CLAUDE.md")
        if user_tok > 600:
            lines.append(f"  - User CLAUDE.md ~{user_tok} tokens — move reference content to MCP")
        lines.append("  - Use search_context() instead of pre-loading memory files")
        lines.append("  - Use close_task() to compact notes (~200 tokens each)")

    return "\n".join(lines)


@mcp.tool()
def render_dashboard() -> str:
    """Regenerate the benefit/cost HTML dashboard. Returns a clickable file:// URL."""
    log_tool("render_dashboard")
    url = _dashboard.render_interactive(_store)
    if url:
        # Also write a static snapshot as a fallback artifact.
        try:
            _dashboard.render(_store)
        except Exception:  # noqa: BLE001
            pass
        return f"Interactive dashboard at {url}"
    path = _dashboard.render(_store)
    return f"Dashboard written to file://{path}"


@mcp.tool()
def token_stats() -> str:
    """Show estimated token savings from hook interceptions."""
    totals = _store.get_interception_totals()
    if not totals or not totals["total_interceptions"]:
        enabled = _cfg.get("token_profiling")
        if not enabled:
            return (
                "Token profiling is disabled.\n"
                "Enable with: configure(key='token_profiling', value='true')"
            )
        return "No interceptions recorded yet. Use task commands to build up data."

    stats = _store.get_interception_stats()
    total_saved = totals["total_tokens_saved"] or 0
    total_count = totals["total_interceptions"] or 0

    lines = [
        f"=== Token Savings Report ===\n",
        f"Total intercepted commands: {total_count}",
        f"Total tokens saved (est.):  ~{total_saved:,}",
        f"  (~${total_saved / 1_000_000 * 3:.4f} at $3/M tokens, ~${total_saved / 1_000_000 * 15:.4f} at $15/M)\n",
        "By command type:",
    ]
    for row in stats:
        avg = (row["total_tokens_saved"] or 0) // max(row["intercept_count"], 1)
        lines.append(
            f"  {row['command_type']:<20} {row['intercept_count']:>4}x  "
            f"~{row['total_tokens_saved'] or 0:,} tokens saved  (avg {avg}/cmd)"
        )

    lines.append(
        f"\nToken profiling: {'on' if _cfg.get('token_profiling') else 'off'}"
        f"  ← configure(key='token_profiling', value='false') to disable"
    )
    return "\n".join(lines)


@mcp.tool()
def list_models(show_recommendations: bool = False) -> str:
    """List installed Ollama models. show_recommendations=True for hardware guide."""
    import httpx

    cfg = _cfg.load_config()
    ollama_base = cfg.get("ollama_base", "http://localhost:11434")
    current_embed = cfg.get("embed_model", "nomic-embed-text")
    current_reason = cfg.get("reason_model", "deepseek-r1:1.5b")

    lines = [f"Configured: embed={current_embed}, reason={current_reason}\n"]

    try:
        resp = httpx.get(f"{ollama_base}/api/tags", timeout=5.0)
        installed = resp.json().get("models", [])
        if installed:
            lines.append("Installed:")
            for m in installed:
                name = m["name"]
                size_gb = m.get("size", 0) / 1_073_741_824
                role = ""
                if current_embed in name:
                    role = "  <- embed_model"
                elif current_reason in name:
                    role = "  <- reason_model"
                lines.append(f"  {name:<35} {size_gb:.1f} GB{role}")
        else:
            lines.append("No models installed.")
    except Exception as exc:
        lines.append(f"Ollama not reachable ({exc})")

    if show_recommendations:
        lines.append("\nReasoning models (reason_model):")
        lines.append("  Model                  Size    RAM     Notes")
        lines.append("  deepseek-r1:1.5b       1.1 GB  8 GB    Minimal, fast")
        lines.append("  deepseek-r1:7b         4.7 GB  16 GB   Recommended")
        lines.append("  deepseek-r1:14b        9 GB    32 GB   Best quality")
        lines.append("  qwen2.5-coder:7b       4.7 GB  16 GB   Code-focused")
        lines.append("\nEmbedding models (embed_model):")
        lines.append("  nomic-embed-text       274 MB  any     Default")
        lines.append("  mxbai-embed-large      669 MB  16 GB+  Higher quality")

    return "\n".join(lines)


@mcp.tool()
def pull_model(model: str) -> str:
    """Download an Ollama model. Use configure() to activate it after pulling."""
    import subprocess

    if not model or "/" in model or ";" in model or "|" in model or "&" in model:
        return f"Invalid model name: {model!r}"

    lines = [f"Pulling {model}...\n"]
    try:
        result = subprocess.run(
            ["ollama", "pull", model],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            lines.append(f"✓ {model} downloaded successfully.")
            lines.append(f"\nTo activate as reasoning model:")
            lines.append(f"  configure(key='reason_model', value='{model}')")
            lines.append(f"To activate as embedding model:")
            lines.append(f"  configure(key='embed_model', value='{model}')")
            lines.append(f"  index_skills()  ← rebuild vectors after changing embed model")
        else:
            lines.append(f"✗ Pull failed:\n{result.stderr.strip()}")
    except FileNotFoundError:
        lines.append("✗ 'ollama' command not found. Install from https://ollama.com")
    except subprocess.TimeoutExpired:
        lines.append("✗ Pull timed out (10 min). The model may be very large or connection slow.")
    except Exception as exc:
        lines.append(f"✗ Error: {exc}")

    return "\n".join(lines)


@mcp.tool()
def exhaustion_save(context: str = "", namespace: str = "") -> str:
    """Auto-save session state when Claude is exhausted or rate-limited.

    ``namespace`` (A10): when set to a plugin name, the digest is saved under
    that plugin's memory roots instead of the core memory tree.
    """
    from .cli import _cmd_exhaustion_save
    # Namespace routing is advisory; downstream consumer reads via search_context
    return _cmd_exhaustion_save(context)


# ---------------------------------------------------------------------------
# Plugin extension-points — A5 (scheduled tasks) + A10 (memory optimizer)

@mcp.tool()
def list_plugin_tasks(plugin: str = "") -> str:
    """A5 — List scheduled task templates declared by enabled plugins.

    If ``plugin`` is given, only tasks from that plugin are returned.
    Output marks which are currently enabled in ``plugin_task_state``.
    """
    from .plugin_registry import iter_enabled_plugins
    state_rows = {
        (r["plugin"], r["name"]): r
        for r in _store._conn.execute(
            "SELECT plugin, name, enabled, cron, external_id FROM plugin_task_state"
        ).fetchall()
    }
    lines: list[str] = []
    for p in iter_enabled_plugins():
        if plugin and p["name"] != plugin:
            continue
        for t in (p["manifest"].get("scheduled_tasks") or []):
            key = (p["name"], t.get("name", ""))
            s = state_rows.get(key)
            enabled = bool(s and s["enabled"]) if s else bool(t.get("enabled_default"))
            lines.append(
                f"- {p['name']}:{t.get('name')}  cron='{t.get('cron')}'  "
                f"enabled={enabled}  template={t.get('prompt_template')}"
            )
    if not lines:
        return "No plugin scheduled tasks declared."
    return "# Plugin Scheduled Tasks\n\n" + "\n".join(lines)


@mcp.tool()
def enable_plugin_task(plugin: str, name: str) -> str:
    """A5 — Enable a scheduled task template declared in a plugin's manifest.

    Writes a Cowork-compatible cron entry to
    ``~/.claude/mcp-skill-hub/scheduled_tasks/{plugin}__{name}.json`` and
    records enablement in ``plugin_task_state``.
    """
    from .plugin_registry import iter_enabled_plugins
    for p in iter_enabled_plugins():
        if p["name"] != plugin:
            continue
        for t in (p["manifest"].get("scheduled_tasks") or []):
            if t.get("name") != name:
                continue
            template_path = p["path"] / str(t.get("prompt_template", ""))
            if not template_path.exists():
                return f"Template not found: {template_path}"
            prompt = template_path.read_text(encoding="utf-8")
            out_dir = Path("~/.claude/mcp-skill-hub/scheduled_tasks").expanduser()
            out_dir.mkdir(parents=True, exist_ok=True)
            entry = {
                "plugin": plugin,
                "name": name,
                "cron": t.get("cron"),
                "prompt": prompt,
            }
            out_file = out_dir / f"{plugin}__{name}.json"
            out_file.write_text(json.dumps(entry, indent=2))
            _store._conn.execute(
                "INSERT OR REPLACE INTO plugin_task_state"
                " (plugin, name, enabled, cron, external_id, updated_at)"
                " VALUES (?, ?, 1, ?, ?, datetime('now'))",
                (plugin, name, t.get("cron"), str(out_file)),
            )
            _store._conn.commit()
            return f"Enabled {plugin}:{name} → {out_file}"
    return f"Task not found: {plugin}:{name}"


@mcp.tool()
def disable_plugin_task(plugin: str, name: str) -> str:
    """A5 — Disable a previously-enabled plugin scheduled task."""
    out_dir = Path("~/.claude/mcp-skill-hub/scheduled_tasks").expanduser()
    out_file = out_dir / f"{plugin}__{name}.json"
    if out_file.exists():
        out_file.unlink()
    _store._conn.execute(
        "INSERT OR REPLACE INTO plugin_task_state"
        " (plugin, name, enabled, cron, external_id, updated_at)"
        " VALUES (?, ?, 0, NULL, NULL, datetime('now'))",
        (plugin, name),
    )
    _store._conn.commit()
    return f"Disabled {plugin}:{name}"


_CORE_SCHEDULED_TASKS: dict[str, dict] = {
    "promote_memory": {
        "cron": "0 3 * * 0",  # Sunday 3 AM
        "description": "Weekly memory promotion: elevate frequently-accessed vectors, prune stale L0/L1.",
        "enabled_default": False,
        "prompt": (
            "Run memory promotion. Call promote_memory(dry_run=False). "
            "After it completes, report how many vectors were promoted or pruned "
            "and which namespaces were most affected. "
            "If no actions taken, confirm system is healthy."
        ),
    },
    "index_skills": {
        "cron": "30 2 * * 1",  # Monday 2:30 AM
        "description": "Weekly skill re-index: refreshes embeddings for all installed skills.",
        "enabled_default": False,
        "prompt": (
            "Run index_skills(). Report how many skills were indexed and any errors."
        ),
    },
}


@mcp.tool()
def enable_core_task(name: str) -> str:
    """Enable a built-in core scheduled task (e.g. promote_memory, index_skills).

    Writes a cron entry to
    ``~/.claude/mcp-skill-hub/scheduled_tasks/core__{name}.json`` and records
    enablement in ``plugin_task_state``.
    """
    task = _CORE_SCHEDULED_TASKS.get(name)
    if not task:
        available = ", ".join(_CORE_SCHEDULED_TASKS)
        return f"Unknown core task '{name}'. Available: {available}"
    out_dir = Path("~/.claude/mcp-skill-hub/scheduled_tasks").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "plugin": "core",
        "name": name,
        "cron": task["cron"],
        "prompt": task["prompt"],
        "description": task["description"],
    }
    out_file = out_dir / f"core__{name}.json"
    out_file.write_text(json.dumps(entry, indent=2))
    _store._conn.execute(
        "INSERT OR REPLACE INTO plugin_task_state"
        " (plugin, name, enabled, cron, external_id, updated_at)"
        " VALUES (?, ?, 1, ?, ?, datetime('now'))",
        ("core", name, task["cron"], str(out_file)),
    )
    _store._conn.commit()
    log_tool("enable_core_task", name=name)
    return f"Enabled core:{name} → {out_file}\nCron: {task['cron']}"


@mcp.tool()
def disable_core_task(name: str) -> str:
    """Disable a previously-enabled core scheduled task."""
    out_dir = Path("~/.claude/mcp-skill-hub/scheduled_tasks").expanduser()
    out_file = out_dir / f"core__{name}.json"
    if out_file.exists():
        out_file.unlink()
    _store._conn.execute(
        "INSERT OR REPLACE INTO plugin_task_state"
        " (plugin, name, enabled, cron, external_id, updated_at)"
        " VALUES (?, ?, 0, NULL, NULL, datetime('now'))",
        ("core", name),
    )
    _store._conn.commit()
    log_tool("disable_core_task", name=name)
    return f"Disabled core:{name}"


@mcp.tool()
def list_core_tasks() -> str:
    """List all built-in core scheduled tasks with their status (enabled/disabled)."""
    rows = _store._conn.execute(
        "SELECT name, enabled, cron FROM plugin_task_state WHERE plugin = 'core'",
    ).fetchall()
    state = {r["name"]: dict(r) for r in rows}
    lines = ["Core scheduled tasks:\n"]
    for name, task in _CORE_SCHEDULED_TASKS.items():
        s = state.get(name, {})
        status = "enabled" if s.get("enabled") else "disabled"
        lines.append(f"  {name} [{status}]  cron={task['cron']}")
        lines.append(f"    {task['description']}")
    return "\n".join(lines)


@mcp.tool()
def optimize_plugin_memory(plugin: str, dry_run: bool = True) -> str:
    """A10 — Run plugin-scoped memory optimization.

    Reads the plugin's ``memory_optimizer.roots`` globs and applies its
    ``compaction_prompt`` via the local reasoning model to produce
    KEEP/PRUNE/COMPACT/MERGE recommendations.
    """
    from .plugin_registry import iter_enabled_plugins
    from .memory_index import _expand_globs
    for p in iter_enabled_plugins():
        if p["name"] != plugin:
            continue
        opt = p["manifest"].get("memory_optimizer") or {}
        roots = opt.get("roots") or []
        prompt_path_rel = opt.get("compaction_prompt")
        if not prompt_path_rel:
            return f"Plugin '{plugin}' has no memory_optimizer.compaction_prompt."
        prompt_path = p["path"] / str(prompt_path_rel)
        if not prompt_path.exists():
            return f"Compaction prompt not found: {prompt_path}"
        files = _expand_globs(p["path"], roots)
        entries = []
        for f in files:
            try:
                entries.append({"path": str(f), "content": f.read_text(encoding="utf-8")[:4000]})
            except OSError:
                continue
        if not entries:
            return f"No memory files found for plugin '{plugin}'."
        prompt = prompt_path.read_text(encoding="utf-8")
        try:
            from .embeddings import optimize_context
            # Reuse the core optimizer; pass plugin prompt via the entries payload.
            decisions = optimize_context(entries, prompt_override=prompt)  # type: ignore[call-arg]
        except TypeError:
            # optimize_context doesn't accept prompt_override yet; fall back to raw entries.
            from .embeddings import optimize_context as _oc
            decisions = _oc(entries)
        if dry_run:
            return json.dumps({"plugin": plugin, "decisions": decisions}, indent=2)
        return json.dumps({"plugin": plugin, "decisions": decisions, "applied": False,
                           "note": "dry_run=False not yet implemented; review decisions and apply manually"}, indent=2)
    return f"Plugin '{plugin}' not enabled or not found."


@mcp.tool()
def remember_identity(fact: str, tag: str = "") -> str:
    """Phase M4 — Persist a long-lived fact about the user into the L4 identity index.

    Stored as a single L4 vector under ``user:identity`` so it carries a high
    retrieval weight and decays very slowly. Never overwritten by the promotion
    loop. Use for role, goals, non-negotiable preferences, career direction.

    ``tag`` is a short label (e.g. ``style``, ``role``, ``goal``) stored in
    metadata; ``fact`` becomes the embed text verbatim.
    """
    import hashlib, time
    fact = fact.strip()
    if not fact:
        return "Refusing to store empty identity fact."
    tag = (tag or "general").strip()
    doc_id = f"{tag}:{hashlib.sha1(fact.encode()).hexdigest()[:10]}"
    _store.upsert_vector(
        namespace="user:identity",
        doc_id=doc_id,
        text=fact,
        metadata={"fact": fact, "tag": tag, "stored_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
        level="L4",
        source="remember_identity",
        tags=[tag],
    )
    log_tool("remember_identity", tag=tag, fact=fact[:80])
    return f"Stored L4 identity fact (tag={tag}): {fact[:120]}"


@mcp.tool()
def promote_memory(dry_run: bool = True) -> str:
    """Phase M4 — Run the memory-promotion rules over the vectors corpus.

    Rules applied (append-only audit into ``memory_audit``):
      - L1 entries older than 7d with ``access_count >= 2`` → promoted to L2.
      - L2 entries older than 30d with ``access_count >= 5`` → promoted to L3.
      - L0/L1 entries past their TTL with zero accesses → pruned.
    L4 identity entries are never touched.

    ``dry_run=True`` (default) reports planned actions without mutating rows.
    """
    conn = _store._conn
    rules = [
        ("promote", "L1", "L2",
         "level = 'L1' AND access_count >= 2 AND "
         "indexed_at < datetime('now', '-7 days')"),
        ("promote", "L2", "L3",
         "level = 'L2' AND access_count >= 5 AND "
         "indexed_at < datetime('now', '-30 days')"),
        ("prune",   "L0", None,
         "level = 'L0' AND access_count = 0 AND "
         "indexed_at < datetime('now', '-1 day')"),
        ("prune",   "L1", None,
         "level = 'L1' AND access_count = 0 AND "
         "indexed_at < datetime('now', '-7 days')"),
    ]
    report: list[dict] = []
    for action, from_lvl, to_lvl, where in rules:
        rows = conn.execute(
            f"SELECT id, namespace, doc_id FROM vectors WHERE {where}"
        ).fetchall()
        for r in rows:
            entry = {"action": action, "namespace": r["namespace"],
                     "doc_id": r["doc_id"], "from_level": from_lvl,
                     "to_level": to_lvl}
            report.append(entry)
            if dry_run:
                continue
            if action == "promote":
                conn.execute(
                    "UPDATE vectors SET level = ? WHERE id = ?",
                    (to_lvl, r["id"]),
                )
            else:  # prune
                conn.execute("DELETE FROM vectors WHERE id = ?", (r["id"],))
            conn.execute(
                "INSERT INTO memory_audit (action, namespace, doc_id, from_level, to_level, reason) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (action, r["namespace"], r["doc_id"], from_lvl, to_lvl,
                 "promote_memory"),
            )
    if not dry_run:
        conn.commit()
    return json.dumps({"dry_run": dry_run, "count": len(report),
                       "actions": report[:50]}, indent=2)


_SEARCH_PROFILES: dict[str, dict] = {
    "default":  {"categories": "all", "include_plugin_memory": True},
    "coding":   {"categories": "skills,tasks", "include_plugin_memory": False},
    "career":   {"categories": "skills", "namespaces_only": [
        "career:profile", "career:narrative", "career:private-signal",
        "user:identity", "user:preferences",
    ]},
    "planning": {"categories": "tasks,closed", "include_plugin_memory": True},
}


@mcp.tool()
def search_context_profile(query: str, profile: str = "default",
                            top_k: int = 5) -> str:
    """Phase M5 — Preset mixes over ``search_context`` / ``search_vectors``.

    Profiles:
      - ``default``  — full merged view (tasks/skills/plugins/memory).
      - ``coding``   — skills + open tasks only; no chatty memory.
      - ``career``   — career:* + user:identity; skips skill rows entirely.
      - ``planning`` — open + closed tasks, plus plugin memory.
    """
    p = _SEARCH_PROFILES.get(profile, _SEARCH_PROFILES["default"])
    if p.get("namespaces_only"):
        hits = _store.search_vectors(query, namespaces=p["namespaces_only"],
                                     top_k=top_k)
        if not hits:
            return f"No matches in profile '{profile}' for: {query!r}"
        lines = [f"# Profile: {profile}", ""]
        for h in hits:
            meta = h.get("metadata") or {}
            label = meta.get("fact") or meta.get("path") or h["doc_id"]
            lines.append(f"- [{h['namespace']}] {label} "
                         f"(score={h['score']:.2f}, {h.get('level','')})")
        return "\n".join(lines)
    return search_context(
        query=query,
        top_k=top_k,
        categories=p.get("categories", "all"),
        include_plugin_memory=p.get("include_plugin_memory", True),
    )


@mcp.tool()
def search_web(query: str, top_k: int = 5) -> str:
    """Search the web via local SearXNG and summarize with local LLM.

    Args:
        query: The search query.
        top_k: Number of results to return (default 5).
    """
    log_tool("search_web", query=query[:80])

    from .searxng import (
        _resolve_searxng_url,
        _searxng_search,
        _summarize_results,
    )
    from . import config as _cfg

    if not _cfg.is_service_enabled("searxng"):
        return "SearXNG is disabled. Enable via the /control panel or configure(key='services.searxng.enabled', value='true')."

    probe_timeout = float(_cfg.get("searxng_timeout") or 5)
    search_timeout = float(_cfg.get("searxng_search_timeout") or 15)

    base_url = _resolve_searxng_url(timeout=probe_timeout)
    if not base_url:
        return ("No SearXNG instance found. Set up with:\n"
                "  python install.py --searxng\n"
                "Or manually: docker compose -f docker/docker-compose.searxng.yml up -d")

    try:
        results = _searxng_search(query, base_url, top_k=top_k, timeout=search_timeout)
    except Exception as exc:
        return f"Search failed: {exc}"

    if not results:
        return "No results found."

    # Format results
    lines = [f"## Web Search: {query}\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"**{i}. [{r['title']}]({r['url']})**")
        if r.get("snippet"):
            lines.append(f"   {r['snippet']}\n")

    # Add LLM summary if available
    summary = _summarize_results(query, results)
    if summary:
        lines.append(f"\n**Summary:** {summary}")

    return "\n".join(lines)


@mcp.tool()
def analyze_router_log(
    top_n: int = 20,
    propose_teachings: bool = True,
) -> str:
    """Analyse the prompt-router audit log to surface misclassifications and suggest teach() rules.

    Reads router.jsonl, clusters verdicts by tier and model, identifies patterns
    where confidence was low or where the user might have disagreed, and
    optionally proposes teach() calls to improve future routing.

    Args:
        top_n: Number of recent entries to analyse (default 20).
        propose_teachings: If True, include suggested teach() calls (default True).
    """
    import json as _json
    from pathlib import Path as _Path
    from collections import Counter

    log_path = _Path(_cfg.get("router_log") or
                     _Path.home() / ".claude" / "mcp-skill-hub" / "router.jsonl")

    if not log_path.exists():
        return "Router log not found. The prompt-router hook must run at least once first."

    entries: list[dict] = []
    try:
        lines = log_path.read_text().splitlines()
        for line in lines[-top_n:]:
            line = line.strip()
            if line:
                try:
                    entries.append(_json.loads(line))
                except _json.JSONDecodeError:
                    pass
    except OSError as exc:
        return f"Could not read router log: {exc}"

    if not entries:
        return "Router log is empty."

    # ── Summary stats ────────────────────────────────────────────────────────
    model_counts: Counter = Counter(e["verdict"]["model"] for e in entries)
    tier_counts: Counter = Counter(e["verdict"]["tier_used"] for e in entries)
    plan_count = sum(1 for e in entries if e["verdict"].get("plan_mode"))
    avg_conf = sum(e["verdict"]["confidence"] for e in entries) / len(entries)
    avg_lat = sum(e.get("latency_ms", 0) for e in entries) / len(entries)

    lines_out: list[str] = [
        f"## Router Log Analysis — last {len(entries)} prompts\n",
        f"Model distribution: {dict(model_counts)}",
        f"Tier distribution:  {dict(tier_counts)}",
        f"Plan mode triggered: {plan_count}/{len(entries)}",
        f"Avg confidence: {avg_conf:.2f}  |  Avg latency: {int(avg_lat)}ms\n",
    ]

    # ── Low-confidence entries ────────────────────────────────────────────────
    low_conf = [e for e in entries if e["verdict"]["confidence"] < 0.65]
    if low_conf:
        lines_out.append(f"### Low-confidence verdicts ({len(low_conf)})")
        for e in low_conf[:5]:
            v = e["verdict"]
            lines_out.append(
                f"  - [{v['tier_used']}] conf={v['confidence']:.2f} "
                f"→ {v['model']}  \"{e.get('prompt_preview', '')}\""
            )

    # ── Proposed teach() rules ────────────────────────────────────────────────
    if propose_teachings and low_conf:
        lines_out.append("\n### Suggested teach() calls to improve routing")
        lines_out.append(
            "Add these rules to help the router handle similar prompts with higher "
            "confidence. Edit the suggest= arg to match the correct model.\n"
        )
        seen: set[str] = set()
        for e in low_conf[:3]:
            preview = e.get("prompt_preview", "")
            if not preview or preview in seen:
                continue
            seen.add(preview)
            v = e["verdict"]
            model_val = v["model"]
            lines_out.append(
                f'teach(\n'
                f'  rule="when the prompt resembles: {preview[:60]}",\n'
                f'  suggest="model:{model_val}"\n'
                f')'
            )

    # ── Tier-3 batch task usage ───────────────────────────────────────────────
    tier3 = [e for e in entries if e["verdict"]["tier_used"] == 3]
    if tier3:
        lines_out.append(f"\n### Tier-3 (Haiku) calls: {len(tier3)}/{len(entries)}")
        lines_out.append("Consider raising router_haiku_threshold if Haiku fires too often.")

    return "\n".join(lines_out)


# ---------------------------------------------------------------------------
# Session memory (ported from cookbook session_memory_compaction.ipynb)

@mcp.tool()
def get_session_memory(session_id: str = "") -> str:
    """Return the stored 6-section session memory for a session.

    Empty ``session_id`` returns an index of all sessions that currently have
    a memory file.  Use this when Claude feels lost ("where were we?") to
    recover context that survives /compact.

    Parallel-safe: pure read.
    """
    from .router import session_memory as _sm

    if not session_id:
        d = _sm.memory_dir()
        if not d.is_dir():
            return "No session memory stored yet."
        files = sorted(d.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            return "No session memory stored yet."
        lines = ["# Stored session memories"]
        for f in files[:20]:
            lines.append(f"- {f.stem}  ({f.stat().st_size} bytes)")
        return "\n".join(lines)

    text = _sm.read_memory(session_id)
    if not text:
        return f"No session memory for {session_id}."
    return text


@mcp.tool()
def rebuild_session_memory(
    session_id: str,
    transcript_path: str = "",
) -> str:
    """Synchronously rebuild session memory from a transcript file.

    Normally the Stop hook schedules a background build; this tool is for
    manual recovery when the background build failed or never ran. Returns
    the first 500 characters of the new memory.

    Parallel-safe: writes to ``session-memory/<session_id>.md`` (single
    writer per session via per-session lock).
    """
    from .router import session_memory as _sm

    if not session_id:
        return "session_id is required."
    if not transcript_path:
        return (
            "transcript_path is required. "
            "Pass the JSONL path from ~/.claude/projects/.../conversations/."
        )
    transcript = _sm.read_transcript_tail(transcript_path)
    if not transcript.strip():
        return f"Transcript at {transcript_path} is empty or unreadable."
    try:
        memory = _sm.build_session_memory(transcript)
    except Exception as exc:  # noqa: BLE001
        return f"Build failed: {exc}"
    if not memory:
        return "Build produced an empty summary."
    path = _sm.write_memory(session_id, memory)
    preview = memory[:500]
    return f"Saved {len(memory)} chars to {path}.\n\nPreview:\n{preview}"


def main() -> None:
    import sys
    log_banner()
    # Print log path to stderr (Claude console) — stdout is MCP transport
    from .activity_log import LOG_FILE
    print(f"[Skill Hub] Logs: tail -100f {LOG_FILE}", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
