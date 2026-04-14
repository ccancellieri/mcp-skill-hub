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
    EMBED_MODEL, RERANK_MODEL, ollama_available,
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
                              auto_approve: bool | None) -> None:
    """Write active task marker so hooks can read it without touching the DB."""
    try:
        _ACTIVE_TASK_MARKER.parent.mkdir(parents=True, exist_ok=True)
        _ACTIVE_TASK_MARKER.write_text(json.dumps({
            "task_id": task_id,
            "session_id": session_id,
            "title": title,
            "auto_approve": auto_approve,
        }, indent=2))
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

# Warm up the FastAPI dashboard in a daemon thread so it's ready before
# the first close_task / render_dashboard call. Safe no-op if disabled or
# the port is busy.
try:
    _dashboard.render_interactive(_store)
except Exception:  # noqa: BLE001
    pass

# Watchdog auto-reindex — starts silently if watchdog is installed
import atexit as _atexit
from .watcher import start_watcher, stop_watcher as _stop_watcher
_watcher = start_watcher()
if _watcher:
    _atexit.register(_stop_watcher, _watcher)

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
    if not ollama_available(EMBED_MODEL):
        return (
            f"Ollama model '{EMBED_MODEL}' not found. "
            f"Run: ollama pull {EMBED_MODEL}"
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
    if not ollama_available(EMBED_MODEL):
        return f"Ollama model '{EMBED_MODEL}' not found. Run: ollama pull {EMBED_MODEL}"

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
    if not ollama_available(EMBED_MODEL):
        return f"Ollama model '{EMBED_MODEL}' not found. Run: ollama pull {EMBED_MODEL}"

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


@mcp.tool()
def save_task(
    title: str,
    summary: str,
    context: str = "",
    tags: str = "",
) -> str:
    """Save an open task for retrieval in future sessions."""
    log_tool("save_task", title=title, tags=tags)
    if not ollama_available(EMBED_MODEL):
        return f"Ollama model '{EMBED_MODEL}' not found. Run: ollama pull {EMBED_MODEL}"

    vector = embed(f"{title}: {summary}")
    tid = _store.save_task(
        title=title, summary=summary, vector=vector,
        context=context, tags=tags, session_id=_session["id"],
    )
    _write_active_task_marker(tid, _session["id"], title, auto_approve=None)
    return f"Task #{tid} saved (open): \"{title}\"\nWill surface in future search_context() calls."


@mcp.tool()
def close_task(task_id: int, summary: str = "") -> str:
    """Close a task with LLM-compacted summary (~200 tokens)."""
    log_tool("close_task", task_id=task_id)
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

    # Re-embed the compacted summary for better future matching
    compact_vector = embed(f"{digest.get('title', '')}: {digest.get('summary', '')}")

    _store.close_task(task_id, compact_text, compact_vector)
    _clear_active_task_marker(task_id)

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

    return (
        f"Task #{task_id} closed and compacted.\n"
        f"Title: {digest.get('title', 'N/A')}\n"
        f"Summary: {digest.get('summary', 'N/A')}\n"
        f"Tags: {digest.get('tags', 'N/A')}\n"
        f"Decisions: {digest.get('decisions', [])}"
        f"{dash_line}"
    )


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
    # Update active marker if this task is the currently active one.
    try:
        if _ACTIVE_TASK_MARKER.exists():
            cur = json.loads(_ACTIVE_TASK_MARKER.read_text())
            if cur.get("task_id") == task_id:
                _write_active_task_marker(
                    task_id, cur.get("session_id", _session["id"]),
                    cur.get("title", task["title"]), auto_approve=enabled,
                )
        elif task["session_id"] == _session["id"] and task["status"] == "open":
            _write_active_task_marker(
                task_id, task["session_id"], task["title"],
                auto_approve=enabled,
            )
    except (OSError, json.JSONDecodeError):
        pass
    label = "null" if enabled is None else str(bool(enabled)).lower()
    return f"Task #{task_id} auto_approve set to {label}."


@mcp.tool()
def update_task(task_id: int, summary: str = "", context: str = "",
                tags: str = "") -> str:
    """Update an open task with new information."""
    log_tool("update_task", task_id=task_id)
    if not ollama_available(EMBED_MODEL):
        return f"Ollama model '{EMBED_MODEL}' not found."

    vector = None
    if summary:
        task = _store.get_task(task_id)
        if task:
            vector = embed(f"{task['title']}: {summary}")

    if _store.update_task(task_id, summary=summary, context=context,
                          tags=tags, vector=vector):
        return f"Task #{task_id} updated."
    return f"Task #{task_id} not found."


@mcp.tool()
def reopen_task(task_id: int) -> str:
    """Reopen a previously closed task."""
    log_tool("reopen_task", task_id=task_id)
    if _store.reopen_task(task_id):
        return f"Task #{task_id} reopened."
    return f"Task #{task_id} not found."


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
        lines.append(f"  #{r['id']} {state} {r['title']}{tags} — {r['summary'][:80]}...")
    return f"{len(lines)} tasks:\n" + "\n".join(lines)


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
    if not ollama_available(EMBED_MODEL):
        return f"Ollama model '{EMBED_MODEL}' not found."

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
    if not ollama_available(EMBED_MODEL):
        return (
            f"Ollama model '{EMBED_MODEL}' not found. "
            f"Run: ollama pull {EMBED_MODEL}"
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
    if not ollama_available(EMBED_MODEL):
        return f"Ollama model '{EMBED_MODEL}' not found. Run: ollama pull {EMBED_MODEL}"

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

    # Index explicit extra_plugin_dirs entries
    for entry in cfg.get("extra_plugin_dirs", []):
        if not entry.get("enabled", True):
            continue
        source = entry.get("source", "extra")
        base = Path(entry["path"]).expanduser()
        if not base.exists():
            continue
        # Each subdirectory is a plugin; read plugin.json or README.md for description
        for plugin_dir in (d for d in base.iterdir() if d.is_dir()):
            desc = entry.get("description", "")
            for manifest in ("plugin.json", "README.md"):
                mf = plugin_dir / manifest
                if mf.exists():
                    try:
                        text = mf.read_text(encoding="utf-8", errors="replace")
                        if manifest == "plugin.json":
                            import json as _json
                            data = _json.loads(text)
                            desc = data.get("description", desc)
                        else:
                            # First paragraph of README
                            para = re.search(r"\S.{20,}", re.sub(r"^#+.*$", "", text, flags=re.MULTILINE))
                            if para:
                                desc = para.group(0)[:200].strip()
                        break
                    except Exception:
                        pass
            if not desc:
                desc = f"Plugin: {plugin_dir.name}"
            plugin_key = f"{source}:{plugin_dir.name}"
            _index_plugin(plugin_key, plugin_dir.name, desc)

    result = f"Indexed {indexed} plugins."
    if errors:
        result += f"\n\nErrors: " + "; ".join(errors[:5])
    return result


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
    if not SETTINGS_PATH.exists():
        return f"Settings file not found: {SETTINGS_PATH}"

    settings = json.loads(SETTINGS_PATH.read_text())
    plugins: dict = settings.setdefault("enabledPlugins", {})

    # Resolve partial name match (e.g. "superpowers" → "superpowers@...")
    if plugin_name not in plugins:
        matches = [k for k in plugins if plugin_name in k]
        if len(matches) == 1:
            plugin_name = matches[0]
        elif len(matches) > 1:
            return (
                f"Ambiguous: {matches}. "
                f"Provide full key like '{matches[0]}'."
            )
        else:
            if enabled:
                plugin_name_full = f"{plugin_name}@claude-plugins-official"
                plugins[plugin_name_full] = True
                SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
                return f"Added '{plugin_name_full}' as enabled (restart to apply)."
            return f"Plugin '{plugin_name}' not found in settings."

    plugins[plugin_name] = enabled
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
    state = "enabled" if enabled else "disabled"
    return f"Plugin '{plugin_name}' {state}. Restart Claude Code to apply."


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
    """Analyze memory files with local LLM. Recommends KEEP/PRUNE/COMPACT/MERGE. dry_run=True for report only.
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

    reason_model = str(_cfg.get("reason_model"))
    if not ollama_available(reason_model):
        return (
            f"Reason model '{reason_model}' not available.\n"
            f"Run: ollama pull {reason_model}"
        )

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
        f"with {reason_model}...\n",
    ]

    # Call local LLM
    results = optimize_context(entries, model=reason_model)

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

    if not _cfg.get("searxng_enabled"):
        return "SearXNG is disabled. Enable with: configure(key='searxng_enabled', value='true')"

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


def main() -> None:
    import sys
    log_banner()
    # Print log path to stderr (Claude console) — stdout is MCP transport
    from .activity_log import LOG_FILE
    print(f"[Skill Hub] Logs: tail -100f {LOG_FILE}", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
