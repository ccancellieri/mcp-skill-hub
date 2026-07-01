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
    embed_unavailable_reason,
    _generate,
)
from .indexer import index_all
from .activity_log import log_tool, log_llm, log_banner
from .resource_monitor import should_run_llm, snapshot
from .store import SkillStore, get_store
from .compression import maybe_compress
from . import dashboard as _dashboard
from .capabilities import requires_capability


def _get_cpu_info() -> int:
    """Get CPU core count for display."""
    import os
    return os.cpu_count() or 0

SETTINGS_PATH = Path.home() / ".claude" / "settings.json"

_store = get_store()

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

# M2 W1 — background prune at startup (off critical path, same as startup_align).
import threading as _threading  # noqa: E402

try:
    _t_prune = _threading.Thread(
        target=lambda: _store.events_prune(),
        name="events-prune-startup",
        daemon=True,
    )
    _t_prune.start()
except Exception:  # noqa: BLE001
    pass

# Continuous memory sweep — opt-in background promote pass (default OFF).
try:
    from .continuous_sweep import start as _start_continuous_sweep
    _start_continuous_sweep()
except Exception:  # noqa: BLE001
    pass

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


# --- M2 W3: uniform tool envelope -----------------------------------------
# Every @mcp.tool() registration is transparently wrapped with tool_envelope
# so each invocation produces a ToolResult (stdout / structured / error /
# elapsed_ms / events_emitted). The wire-level callable still returns the
# plain str / dict that FastMCP serializes — no client change.
from .envelope import tool_envelope as _tool_envelope  # noqa: E402

_orig_mcp_tool = mcp.tool


def _mcp_tool_with_envelope(*args, **kwargs):
    raw_decorator = _orig_mcp_tool(*args, **kwargs)

    def deco(fn):
        return raw_decorator(_tool_envelope(fn))

    return deco


mcp.tool = _mcp_tool_with_envelope  # type: ignore[assignment]


# --- M2 W1: wire emit hook into envelope -----------------------------------
# Set up a closure that reads the current session id from _session["id"] and
# calls _store.append_event.  The hook is non-fatal by design: a failure here
# must never break the tool call that triggered it.
from .envelope import set_emit_hook as _set_emit_hook  # noqa: E402


def _make_emit_hook():
    def _emit(kind: str, tool_name: str | None, payload: dict) -> "int | None":
        try:
            sid = _session.get("id", "")
            return _store.append_event(
                session_id=sid,
                kind=kind,
                tool_name=tool_name,
                payload=payload,
            )
        except Exception:  # noqa: BLE001
            return None
    return _emit


_set_emit_hook(_make_emit_hook())

# Emit session_start for this process boot (session already in _session["id"]).
try:
    _store.append_event(
        session_id=_session["id"],
        kind="session_start",
        tool_name=None,
        payload={"source": "server_boot"},
    )
except Exception:  # noqa: BLE001
    pass


# ---------------------------------------------------------------------------
# Search & Load


# Transient in-process state so record_feedback can omit query when called
# right after search_skills.
_last_search_state: dict = {"query": "", "vector": [], "skills": []}


@mcp.tool()
@requires_capability("embedding")
def search_skills(
    query: str,
    top_k: int = 5,
    use_rerank: bool = False,
) -> str:
    """Semantic skill search. Returns full content of top matches."""
    from .activity_log import LOG_FILE

    log_tool("search_skills", query=query, top_k=top_k, rerank=use_rerank)

    # Degraded-search fallback: when no embedding backend is available, fall
    # back to SQLite FTS5 keyword/BM25 search so callers get *something* useful
    # instead of an error.
    if not embed_available():
        fts_hits = _store.search_skills_text(query, top_k=top_k)
        if not fts_hits:
            return (
                f"<!-- Skill Hub search: query={query!r} top_k={top_k} mode=keyword-fts5 -->\n"
                f"No matching skills found via keyword fallback. "
                f"Start Ollama with '{EMBED_MODEL}', or install "
                f"sentence-transformers for semantic search."
            )
        loaded_ids = [c["id"] for c in fts_hits]
        _last_search_state["query"] = query
        _last_search_state["vector"] = []
        _last_search_state["skills"] = loaded_ids
        for c in fts_hits:
            try:
                _store.log_skill_injection(c["id"], query, _session.get("id"))
            except Exception:
                pass
        max_skill_chars = int(_cfg.get("hook_context_max_skill_chars") or 8000)
        header = "\n".join([
            f"<!-- Skill Hub search: query={query!r} top_k={top_k} mode=keyword-fts5 -->",
            f"<!-- LOADED ({len(fts_hits)}): {', '.join(loaded_ids) or 'none'} -->",
            "<!-- degraded-search: embeddings unavailable, used FTS5 BM25 fallback -->",
        ])
        parts: list[str] = [header]
        for c in fts_hits:
            content = (c["content"] or "").strip()
            if len(content) > max_skill_chars:
                content = content[:max_skill_chars] + "\n\n<!-- truncated -->"
            parts.append(f"<!-- skill: {c['id']} -->\n{content}")
        return "\n\n---\n\n".join(parts)

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
        f"<!-- Skill Hub search: query={query!r} top_k={top_k} mode=vector -->",
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

    return maybe_compress(
        "\n\n---\n\n".join(parts), context=query, site="search_skills"
    )


@mcp.tool()
@requires_capability("embedding")
def suggest_plugins(query: str = "") -> str:
    """Suggest disabled plugins matching the current task."""
    log_tool("suggest_plugins", query=query)

    used_query = query or _last_search_state.get("query", "")
    if not used_query:
        return "Provide a query or call search_skills first."

    # Degraded-search fallback (M1 #8): when no embedding backend is available,
    # use FTS5 BM25 keyword search so callers still get ranked suggestions.
    if not embed_available():
        suggestions = _store.suggest_plugins_text(used_query)
        mode_marker = "<!-- mode=keyword-fts5 (degraded-search: embeddings unavailable) -->"
    else:
        query_vector = embed(used_query) if query else _last_search_state.get("vector", [])
        if not query_vector:
            query_vector = embed(used_query)
        suggestions = _store.suggest_plugins(query_vector)
        mode_marker = "<!-- mode=vector -->"

    if not suggestions:
        return (
            f"{mode_marker}\n"
            f"No plugin suggestions for this query. Add teachings with teach() to improve."
        )

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

    return f"{mode_marker}\nPlugin suggestions for \"{used_query}\":\n\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Learning


@mcp.tool()
@requires_capability("none")
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
@requires_capability("embedding")
def teach(rule: str, suggest: str, cwd: str = "", override_pii: bool = False) -> str:
    """Add a persistent rule mapping task patterns to plugins or skills.

    The PII gate scans `rule` + `suggest` when `cwd` resolves to a repo whose
    ``.skill-hub/policy.yml`` declares ``public: true``. Pass ``override_pii=True``
    to bypass the block (the override is logged to
    ``<repo>/.skill-hub/pii_overrides.log``).
    """
    log_tool("teach", rule=rule, suggest=suggest)

    # M1 PII gate
    _gate_content = "\n".join(filter(None, [rule, suggest]))
    _allowed, _msg = _enforce_pii_gate(
        tool="teach", content=_gate_content, cwd=cwd, override=override_pii,
    )
    if not _allowed:
        return _msg

    if not embed_available():
        return embed_unavailable_reason()

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
@requires_capability("none")
def forget_teaching(teaching_id: int) -> str:
    """Remove a teaching rule by its ID."""
    log_tool("forget_teaching", teaching_id=teaching_id)
    if _store.remove_teaching(teaching_id):
        return f"Teaching #{teaching_id} removed."
    return f"Teaching #{teaching_id} not found."


@mcp.tool()
@requires_capability("none")
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
@requires_capability("none")
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
@requires_capability("none")
def retrieve_compressed(hash: str) -> str:
    """Retrieve the original content behind a ``<<ccr:HASH>>`` compression marker.

    Skill Hub may inject reversible, deterministic compression markers
    (``<<ccr:HASH>>``) when it shrinks large structured tool outputs, logs, or JSON
    before showing them. Call this with the hash to get the full original text back
    on demand.
    """
    log_tool("retrieve_compressed", hash=hash)
    from .compression import retrieve_original

    original = retrieve_original(hash)
    if original is None:
        return (
            f"No stored original for hash '{hash}'. The entry may have expired, "
            f"or deterministic compression is not enabled (install the 'compression' extra)."
        )
    return original


@mcp.tool()
@requires_capability("none")
def ensure_tooling(path: str, init: bool = False, refresh: bool = True) -> str:
    """Probe and optionally provision dev-tooling readiness for a directory path.

    Checks whether the code-graph index for *path* is present and up to date.
    When *refresh* is True (default) and an index already exists, a background
    sync is dispatched (fire-and-forget, non-blocking).
    When *init* is True and no index exists, ``codegraph init`` is run blocking
    — use this only after the user has confirmed they want initialization.

    Idempotent: safe to call multiple times; repeated calls within the sync-TTL
    window are debounced automatically.

    Returns a human-readable readiness summary including the steering directive.
    """
    log_tool("ensure_tooling", path=path, init=init, refresh=refresh)
    try:
        from .orchestrator import ensure_tooling_core
        result = ensure_tooling_core(path, init=init, refresh=refresh)
        status = "present" if result["present"] else "absent"
        freshness = "fresh" if result["fresh"] else "stale"
        action = result["action"]
        directive = result.get("directive", "")
        summary_parts = [
            f"path: {result['path']}",
            f"index: {status} ({freshness})",
            f"action: {action}",
        ]
        if directive:
            summary_parts.append(directive)
        return " | ".join(summary_parts)
    except Exception as exc:  # noqa: BLE001
        return f"ensure_tooling: error probing {path}: {exc}"


@mcp.tool()
@requires_capability("none")
def close_session(summary: str = "", to_wiki: bool = False) -> str:
    """Phase M3 — Close the current session and persist its L1 summary.

    - Writes ``summary`` (plus the session's tracked topic) into the
      ``session:log`` index as an L1 vector so future ``search_context``
      calls can surface "we worked on this before".
    - Flushes any partial tool-chain window into ``habits:tool-chains``.
    - Dispatches the ``on_session_end`` plugin hook (A3) so plugins can
      observe the session boundary.
    - Rotates the in-process session id so subsequent work starts fresh.
    - When ``to_wiki=True``, enqueues the session's memory into the wiki
      ingest approval queue via ``wiki.scan_and_enqueue`` (no token spend;
      human approval is still required before distillation).
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
    # M2 W1 — emit session_end before rotating the session id.
    try:
        _store.append_event(
            session_id=sid,
            kind="session_end",
            tool_name=None,
            payload={"topic": topic, "summary": text},
        )
    except Exception:  # noqa: BLE001
        pass
    # Kick off a background prune (non-blocking — daemon thread, fire-and-forget).
    try:
        import threading as _threading
        _t = _threading.Thread(
            target=lambda: _store.events_prune(),
            name="events-prune",
            daemon=True,
        )
        _t.start()
    except Exception:  # noqa: BLE001
        pass
    try:
        from . import plugin_hooks
        plugin_hooks.dispatch(
            "on_session_end",
            {"session_id": sid, "topic": topic, "summary": text},
        )
    except Exception:  # noqa: BLE001
        pass
    new_sid = str(uuid.uuid4())
    _session["id"] = new_sid
    _session["topic"] = ""
    _session["topic_vector"] = []
    _session["tool_chain"] = []
    # Emit session_start for the fresh session.
    try:
        _store.append_event(
            session_id=new_sid,
            kind="session_start",
            tool_name=None,
            payload={"source": "close_session_rotate"},
        )
    except Exception:  # noqa: BLE001
        pass
    result_msg = f"Session closed → session:log (new id={_session['id'][:8]})"

    # Optional wiki enqueue — auto-select, NOT auto-spend.  Fail-open.
    if to_wiki:
        try:
            from . import wiki as _wiki_mod
            from pathlib import Path as _wpath
            _wiki_root = _wpath(_cfg.get("wiki_root") or
                                _wpath.home() / ".claude" / "mcp-skill-hub" / "wiki")
            enq = _wiki_mod.scan_and_enqueue(_store, _wiki_root)
            result_msg += (
                f"; wiki queue: scanned={enq.get('scanned', 0)} "
                f"enqueued pending={enq.get('pending', 0)}"
            )
        except Exception as _exc:  # noqa: BLE001
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "close_session: wiki enqueue failed: %s", _exc)
            result_msg += "; wiki enqueue failed (logged)"

    return result_msg


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
            generated = _generate(prompt, model=RERANK_MODEL, timeout=15.0, num_predict=30, op="task_title")
            generated = generated.strip().strip('"').strip("'")
            if generated and len(generated) < 120:
                return generated
        except Exception:
            pass
    # Fallback: first non-empty line, capped at 80 chars
    first_line = next((ln.strip() for ln in content.splitlines() if ln.strip()), content)
    return first_line[:80]


def _enforce_pii_gate(
    *, tool: str, content: str, cwd: str = "", project: str = "",
    repo: str = "", override: bool = False,
) -> tuple[bool, str]:
    """Thin wrapper over ``pii_gate.enforce`` for save_task/teach.

    Returns (False, refusal_message) when the destination repo is marked
    ``public: true`` in ``.skill-hub/policy.yml`` and the content contains
    likely-private values. Otherwise returns (True, "").
    """
    from . import pii_gate
    return pii_gate.enforce(
        tool=tool, content=content, cwd=cwd, project=project,
        repo=repo, override=override,
    )


@mcp.tool()
@requires_capability("none")
def save_task(
    title: str,
    summary: str,
    context: str = "",
    tags: str = "",
    project: str = "",
    mode: str = "",
    initial_prompt: str = "",
    cwd: str = "",
    repo: str = "",
    override_pii: bool = False,
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
        repo=<name>             explicit per-repo tag (M3 federation). When
                                empty, auto-derived from the worktree spec or
                                detect_project_from_cwd(cwd). Used by
                                ``list_tasks(repo=...)`` and the dashboard's
                                per-repo grouping.

    Parallel-safe: each call creates an independent row (distinct task IDs).
    Two concurrent calls with identical args produce two separate tasks rather
    than merging -- callers should deduplicate before saving.
    """
    log_tool("save_task", title=title, tags=tags, project=project, mode=mode)

    # M1 PII gate — refuse save when content carries likely-private values
    # and the destination repo is marked public in .skill-hub/policy.yml.
    _gate_content = "\n".join(filter(None, [title, summary, context, tags]))
    _allowed, _msg = _enforce_pii_gate(
        tool="save_task", content=_gate_content,
        cwd=cwd, project=project, repo=repo, override=override_pii,
    )
    if not _allowed:
        return _msg

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
    # M3 — repo auto-capture. Explicit kwarg wins; otherwise prefer the
    # worktree spec's project, then detect_project_from_cwd.
    resolved_repo = (repo or "").strip()
    if not resolved_repo:
        if spec is not None:
            resolved_repo = getattr(spec, "project", "") or ""
        elif resolved_project:
            resolved_repo = resolved_project
        elif cwd:
            try:
                resolved_repo = _wt.detect_project_from_cwd(cwd) or ""
            except Exception:  # noqa: BLE001
                resolved_repo = ""

    # M1 #11 — worktree-aware tasks. When the caller didn't spawn a worktree
    # (spec is None) we still want to record where the task is being authored
    # so future sessions can answer "what tasks belong to this worktree?"
    # without manual grepping. Falls back to whatever the caller passed when
    # git inspection fails (e.g. cwd outside any repo).
    auto_cwd = cwd or ""
    auto_branch = ""
    if spec is not None:
        auto_cwd = spec.worktree_path
        auto_branch = spec.branch
    else:
        # Only inspect git when we have an explicit cwd -- the MCP server runs
        # as a daemon so os.getcwd() is meaningless here.
        if cwd:
            try:
                top, br, _ = _wt.capture_worktree_context(cwd)
                if top:
                    auto_cwd = top
                if br:
                    auto_branch = br
            except Exception:  # noqa: BLE001
                pass

    tid = _store.save_task(
        title=title, summary=summary, vector=vector,
        context=context, tags=tags, session_id=_session["id"],
        cwd=auto_cwd,
        branch=auto_branch,
        worktree=worktree_blob,
        repo=resolved_repo,
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
@requires_capability("none")
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


# ───────────────────────── Fanout (parallel issue dispatch) ──────────────────

@mcp.tool()
@requires_capability("none")
def fanout_issues(
    project: str,
    source: str = "gh",
    filter: str = "",
    limit: int = 0,
    repo: str = "",
    dry_run: bool = False,
    use_llm: bool = True,
) -> str:
    """Fan out N issues into N worktree-bound tasks + an Agent dispatch directive.

    One call prepares the parallel-work scaffolding the active Claude needs to
    dispatch `Agent({...})` calls in a single message.

    Parameters
    ----------
    project: skill-hub project name (resolved under worktree.repo_roots).
    source:  "gh" (default) | "text" | a configured custom adapter name.
    filter:  source-specific filter — for gh, a search query like
             "label:bug is:open"; for text, the raw bullet/numbered list.
    limit:   max issues to fan out (0 → fanout.default_limit, default 3).
    repo:    optional "owner/name" passed to sources that support it (gh).
    dry_run: when True, no worktrees / tasks are created — preview only.
    use_llm: when False, every per-issue prompt uses the deterministic
             fallback template instead of the local LLM.

    Returns a multi-line string: summary, per-task rows, and the directive
    block the active Claude should paste back as ONE message with N Agent
    tool uses (so they run concurrently).
    """
    log_tool("fanout_issues", project=project, source=source,
             limit=limit, dry_run=dry_run)
    from .fanout import fanout as _fanout
    try:
        result = _fanout(
            source,
            filter=filter,
            limit=(limit or None),
            project=project,
            repo=repo,
            dry_run=dry_run,
            use_llm=use_llm,
            store=_store,
        )
    except (ValueError, RuntimeError) as e:
        return f"fanout failed: {e}"
    skipped_line = ""
    if result.skipped:
        rows = "\n".join(f"  - {s['issue']}: {s['reason']}" for s in result.skipped)
        skipped_line = f"\n\nSkipped:\n{rows}"
    return result.directive + skipped_line


@mcp.tool()
@requires_capability("none")
def fanout_status(group_id: str) -> str:
    """Roll up progress for all tasks in a fanout group.

    Shows open/closed counts and per-task one-liners (id, title, status).
    """
    log_tool("fanout_status", group_id=group_id)
    rows = _store.list_tasks(status="all", tag=f"fanout:{group_id}")
    if not rows:
        return f"fanout group {group_id}: no tasks found."
    open_n = sum(1 for r in rows if r["status"] == "open")
    closed_n = sum(1 for r in rows if r["status"] == "closed")
    lines = [
        f"fanout group `{group_id}` — {len(rows)} tasks "
        f"({open_n} open, {closed_n} closed)",
    ]
    for r in rows:
        lines.append(f"  #{r['id']} [{r['status']}] {r['title']}")
    return "\n".join(lines)


@mcp.tool()
@requires_capability("none")
def fanout_close(group_id: str, summary: str = "") -> str:
    """Close every open task in a fanout group.

    Uses a short shared summary (or a placeholder) to avoid running the
    per-task LLM compactor N times. Worktrees stay on disk so each task
    can still be resumed if needed.
    """
    log_tool("fanout_close", group_id=group_id)
    rows = _store.list_tasks(status="open", tag=f"fanout:{group_id}")
    if not rows:
        return f"fanout group {group_id}: no open tasks to close."
    digest = (summary or f"fanout group {group_id} closed in bulk").strip()
    closed: list[int] = []
    for r in rows:
        try:
            ok = _store.close_task(r["id"], compact=digest)
            if ok:
                closed.append(r["id"])
        except Exception:  # noqa: BLE001
            continue
    return (
        f"fanout group `{group_id}`: closed {len(closed)} task(s) "
        f"(ids: {', '.join(str(i) for i in closed) or 'none'})"
    )


@mcp.tool()
@requires_capability("none")
def fanout_cleanup(
    group_id: str,
    close_open_tasks: bool = True,
    remove_worktrees: bool = True,
    delete_branches: bool = True,
    summary: str = "",
) -> str:
    """Tear down every task + worktree + branch in a fanout group.

    Bulk inverse of fanout_issues. Closes any still-open tasks (with a short
    shared summary), runs ``git worktree remove --force`` on each task's
    recorded worktree path, and deletes the ``cc/<slug>`` branch. Idempotent.
    """
    from .fanout.coordinator import fanout_cleanup as _cleanup

    log_tool("fanout_cleanup", group_id=group_id)
    res = _cleanup(
        group_id,
        close_open_tasks=close_open_tasks,
        remove_worktrees=remove_worktrees,
        delete_branches=delete_branches,
        summary=summary,
        store=_store,
    )
    lines = [
        f"fanout group `{group_id}` cleanup:",
        f"  closed tasks:     {len(res.closed_task_ids)} "
        f"({', '.join(str(i) for i in res.closed_task_ids) or 'none'})",
        f"  worktrees gone:   {len(res.removed_worktrees)}",
        f"  branches deleted: {len(res.deleted_branches)}",
    ]
    if res.skipped:
        lines.append(f"  skipped:          {len(res.skipped)}")
        for s in res.skipped[:5]:
            lines.append(f"    - {s}")
    return "\n".join(lines)


# ─────────────────────── Worktree pre-flight (M3-1) ─────────────────────────

@mcp.tool()
@requires_capability("none")
def worktree_preflight(
    issue_number: int,
    project: str,
    repo: str = "",
) -> str:
    """Pre-flight collision check before starting a worktree-bound issue task.

    Sub-second three-axis check that encodes the worktree-naming-collision
    rule as a callable tool rather than a memory rule re-read every session:

    1. Local worktrees under ``<repo>/.claude/worktrees/issue-<N>-*``
    2. Local branches matching ``cc/issue-<N>-*``
    3. Open GitHub PRs whose head branch starts with ``cc/issue-<N>-``
       (best-effort via ``gh``; skipped with a warning if ``gh`` is
       missing / unauthenticated)

    Parameters
    ----------
    issue_number: positive GitHub issue id.
    project:      skill-hub project name (resolved under
                  ``worktree.repo_roots``); the *local* repo whose
                  worktrees / branches we inspect.
    repo:         optional ``owner/name`` passed to ``gh`` for the issue
                  metadata + PR lookup. Always set this from the MCP
                  daemon — its cwd is not the user's shell cwd.

    Returns a single string: either "safe to start" or "collision
    detected" plus per-axis details and any ``gh`` warnings.
    """
    from .worktree_preflight import preflight as _preflight, format_result
    log_tool("worktree_preflight", issue_number=issue_number,
             project=project, repo=repo)
    try:
        res = _preflight(issue_number, project=project, repo=repo)
    except (ValueError, RuntimeError) as e:
        return f"worktree_preflight failed: {e}"
    return format_result(res)


@mcp.tool()
@requires_capability("none")
def federation_view(remote_db_path: str, alias: str = "remote") -> str:
    """M4-3 federation-lite — open a peer host's skill-hub DB read-only.

    Attaches ``remote_db_path`` to the local SQLite connection, reports a
    summary of what's visible there (task / event counts, distinct node_ids),
    then detaches. Intended for multi-host setups where the DB file is shared
    via Syncthing / rsync / git-annex — no network protocol involved.

    Use cases:
    - "Whose tasks are these?" — list node_ids in the synced replica.
    - "Did host X record events I haven't seen?" — count by node_id.
    - Cross-host queries can be built manually by ATTACHing in a session.
    """
    log_tool("federation_view", remote_db_path=remote_db_path, alias=alias)
    try:
        info = _store.federation_view(remote_db_path, alias=alias)
    except FileNotFoundError as exc:
        return f"federation_view error: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"federation_view failed: {exc.__class__.__name__}: {exc}"

    nodes = info["remote_nodes"]
    nodes_line = ", ".join(nodes) if nodes else "(none)"
    return (
        f"federation_view: {info['remote_path']}\n"
        f"  alias:        {info['alias']}\n"
        f"  local node:   {info['local_node']}\n"
        f"  remote nodes: {nodes_line}\n"
        f"  tasks  local={info['tasks']['local']}  remote={info['tasks']['remote']}\n"
        f"  events local={info['events']['local']}  remote={info['events']['remote']}\n"
        f"  schemas: tasks={info['schemas']['tasks_remote']} "
        f"events={info['schemas']['events_remote']}"
    )


@mcp.tool()
@requires_capability("llm")
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
        ass_count = result.get("assumptions_count", 0)
        inbox_path = result.get("inbox")
        ass_line = (
            f"\nAssumptions surfaced: {ass_count} (appended to {inbox_path})"
            if ass_count and inbox_path else ""
        )
        pruned = result.get("backups_pruned", 0)
        prune_line = f"\nOld backups pruned: {pruned}" if pruned else ""
        return (
            f"Wrote: {result.get('wrote')}\n"
            f"Backup: {result.get('backup') or '(none — first write)'}\n"
            f"Delta: {result.get('delta_chars'):+d} chars\n"
            f"Memory files folded in: {len(result.get('memory_files_considered', []))}"
            f"{ass_line}"
            f"{prune_line}"
        )
    return f"{result.get('status', 'unknown')}: {result.get('reason', '')}"


@mcp.tool()
@requires_capability("none")
def export_policies(
    project_root: str,
    output_file: str = ".skill-hub/POLICY.md",
    dry_run: bool = False,
    force: bool = False,
) -> str:
    """Render feedback_* memory files as a per-repo POLICY.md.

    Reads the project's auto-memory ``feedback_*.md`` rules and writes a
    paraphrased, in-repo policy document at ``<project_root>/<output_file>``.
    Path references to ``~/.claude/`` / ``.claude/`` are scrubbed so the
    rendered file is safe to commit.

    Idempotent: rerun with no newer feedback files returns ``noop``. Use
    ``force=True`` to rewrite anyway. ``dry_run=True`` returns the rendered
    Markdown without touching disk.
    """
    log_tool("export_policies", project_root=project_root, dry_run=dry_run, force=force)
    from .policy_export import export_policies as _ep
    result = _ep(
        project_root=project_root,
        output_file=output_file,
        dry_run=dry_run,
        force=force,
    )
    status = result.get("status", "unknown")
    if status == "dry_run":
        files = result.get("feedback_files_considered", [])
        rendered = result.get("rendered", "")
        return (
            f"DRY RUN — {len(rendered)} chars rendered from {len(files)} feedback files.\n"
            f"--- POLICY.md preview ---\n{rendered}"
        )
    if status == "written":
        return (
            f"Wrote: {result.get('wrote')}\n"
            f"Backup: {result.get('backup') or '(none — first write)'}\n"
            f"Delta: {result.get('delta_chars'):+d} chars\n"
            f"Feedback files folded in: {len(result.get('feedback_files_considered', []))}"
        )
    if status == "empty":
        return (
            f"Wrote placeholder POLICY.md ({result.get('wrote')}). "
            "No feedback_* memory files found yet."
        )
    return f"{status}: {result.get('reason', '')}"


@mcp.tool()
@requires_capability("none")
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
@requires_capability("none")
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


# ──────────────────────── M1 — task claims layer ─────────────────────────────
#
# Pure-SQLite ownership transitions so multiple Claude Code sessions can
# coordinate work-item ownership without an LLM round-trip. Existing
# single-session task flow is unaffected when ``claimed_by`` stays NULL.


@mcp.tool()
@requires_capability("none")
def claim_task(
    task_id: int,
    agent_id: str,
    stealable_after_sec: int = 0,
) -> str:
    """Claim an unclaimed open task for ``agent_id``.

    Args:
        task_id:              The task to claim.
        agent_id:             Stable identifier for the claiming session /
                              agent. Required.
        stealable_after_sec:  When > 0, allow ``steal_task`` to seize the
                              claim once this many seconds have elapsed.
                              Use 0 to keep the claim non-stealable.

    Returns the claim_token on success, or a "rejected" message when the
    task is already claimed / closed / missing.
    """
    log_tool("claim_task", task_id=task_id, agent_id=agent_id,
             stealable_after_sec=stealable_after_sec)
    if not agent_id:
        return "claim rejected: agent_id required."
    task = _store.get_task(task_id)
    if task is None:
        return f"Task #{task_id} not found."
    after = stealable_after_sec if stealable_after_sec > 0 else None
    token = _store.claim_task(task_id, agent_id, stealable_after_sec=after)
    if token is None:
        claim = _store.get_task_claim(task_id) or {}
        holder = claim.get("claimed_by")
        if holder:
            return (
                f"Task #{task_id} already claimed by {holder} "
                f"(since {claim.get('claimed_at')})."
            )
        return f"Task #{task_id} cannot be claimed (closed or missing)."
    return f"Task #{task_id} claimed by {agent_id} (token={token[:8]})."


@mcp.tool()
@requires_capability("none")
def handoff_task(
    task_id: int,
    to_agent: str,
    from_agent: str = "",
    stealable_after_sec: int = 0,
) -> str:
    """Hand off a claimed task from its current owner to ``to_agent``.

    Args:
        task_id:              The task being handed off.
        to_agent:             New owner agent_id. Required.
        from_agent:           When set, the handoff only succeeds if it
                              matches the current owner. Empty string skips
                              the check (admin/hub paths).
        stealable_after_sec:  Reset the stealable window for the new owner.
    """
    log_tool("handoff_task", task_id=task_id, to_agent=to_agent,
             from_agent=from_agent)
    if not to_agent:
        return "handoff rejected: to_agent required."
    task = _store.get_task(task_id)
    if task is None:
        return f"Task #{task_id} not found."
    after = stealable_after_sec if stealable_after_sec > 0 else None
    token = _store.handoff_task(
        task_id, to_agent,
        from_agent=from_agent or None,
        stealable_after_sec=after,
    )
    if token is None:
        claim = _store.get_task_claim(task_id) or {}
        holder = claim.get("claimed_by")
        if holder is None:
            return f"Task #{task_id} is unclaimed — call claim_task instead."
        if from_agent and holder != from_agent:
            return (
                f"Task #{task_id} handoff rejected: held by {holder!r}, "
                f"not {from_agent!r}."
            )
        return f"Task #{task_id} handoff rejected."
    return (
        f"Task #{task_id} handed off to {to_agent} (token={token[:8]})."
    )


@mcp.tool()
@requires_capability("none")
def steal_task(
    task_id: int,
    new_agent_id: str,
    stealable_after_sec: int = 0,
) -> str:
    """Steal a stale claim whose ``stealable_at`` has elapsed.

    Fails if the task is unclaimed, still inside the stealable window, or
    was claimed without a stealable_at expiry.
    """
    log_tool("steal_task", task_id=task_id, new_agent_id=new_agent_id)
    if not new_agent_id:
        return "steal rejected: new_agent_id required."
    task = _store.get_task(task_id)
    if task is None:
        return f"Task #{task_id} not found."
    after = stealable_after_sec if stealable_after_sec > 0 else None
    token = _store.steal_task(
        task_id, new_agent_id, stealable_after_sec=after,
    )
    if token is None:
        claim = _store.get_task_claim(task_id) or {}
        if claim.get("claimed_by") is None:
            return f"Task #{task_id} is unclaimed — call claim_task instead."
        if not claim.get("stealable_at"):
            return (
                f"Task #{task_id} steal rejected: claim is non-stealable. "
                f"Owner must release_task or handoff_task."
            )
        return (
            f"Task #{task_id} steal rejected: not yet stealable "
            f"(stealable_at={claim.get('stealable_at')})."
        )
    return f"Task #{task_id} stolen by {new_agent_id} (token={token[:8]})."


@mcp.tool()
@requires_capability("none")
def release_task(task_id: int, agent_id: str = "") -> str:
    """Release the claim on a task (clears ``claimed_by``).

    When ``agent_id`` is set, the release only fires if it matches the
    current owner. Empty string allows admin-style force-release.
    """
    log_tool("release_task", task_id=task_id, agent_id=agent_id)
    task = _store.get_task(task_id)
    if task is None:
        return f"Task #{task_id} not found."
    ok = _store.release_task(task_id, agent_id=agent_id or None)
    if not ok:
        claim = _store.get_task_claim(task_id) or {}
        holder = claim.get("claimed_by")
        if holder is None:
            return f"Task #{task_id} was already unclaimed."
        if agent_id and holder != agent_id:
            return (
                f"Task #{task_id} release rejected: held by {holder!r}, "
                f"not {agent_id!r}."
            )
        return f"Task #{task_id} release rejected."
    return f"Task #{task_id} released."


@mcp.tool()
@requires_capability("none")
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
            new_summary = summary or task["summary"] or ""
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
@requires_capability("none")
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
@requires_capability("none")
def list_tasks(status: str = "open", repo: str = "",
               group_by_repo: bool = False,
               worktree_current: bool = False,
               cwd: str = "") -> str:
    """List tasks. status: open (default), closed, or all.

    Args:
        status: open | closed | all.
        repo:   M3 federation filter — only return tasks tagged with this
                repo (auto-captured by ``save_task``). Empty string returns
                all repos. Use this to answer "what tasks are open for repo
                X right now?" without manual grepping.
        group_by_repo: when True, render output grouped under per-repo
                headings (matches the dashboard layout). Otherwise output is
                a flat list ordered by ``updated_at DESC``.
        worktree_current: M1 #11 filter — only return tasks whose recorded
                ``cwd`` matches the current worktree's ``git rev-parse
                --show-toplevel``. Requires ``cwd`` to be set (the MCP
                server runs as a daemon so it cannot trust ``os.getcwd()``).
        cwd:    explicit working directory used to resolve ``worktree_current``.
                Ignored when ``worktree_current=False``.
    """
    log_tool("list_tasks", status=status, repo=repo,
             group_by_repo=group_by_repo,
             worktree_current=worktree_current)
    wt_filter: str | None = None
    if worktree_current:
        from . import worktree as _wt
        if not cwd:
            return ("list_tasks(worktree_current=True) requires cwd= to be "
                    "passed -- the MCP server runs as a daemon and cannot "
                    "trust os.getcwd().")
        wt_filter = _wt.git_toplevel(cwd)
        if not wt_filter:
            return (f"cwd={cwd!r} is not inside a git repository; "
                    "worktree_current filter unavailable.")
    rows = _store.list_tasks(status, repo=(repo or None),
                             worktree_path=wt_filter)
    if not rows:
        scope = f"repo={repo!r} " if repo else ""
        if worktree_current:
            return (f"No {status} tasks for current worktree "
                    f"({wt_filter}).")
        return f"No {status} tasks{(' for ' + scope).rstrip() if repo else ''}."

    def _fmt(r) -> str:
        state = f"[{r['status'].upper()}]"
        tags = f" ({r['tags']})" if r['tags'] else ""
        try:
            color = r["color"]
        except (IndexError, KeyError):
            color = None
        glyph = _COLOR_GLYPH.get(color or "", " ")
        return (
            f"  {glyph} #{r['id']} {state} {r['title']}{tags} — "
            f"{r['summary'][:80]}..."
        )

    if not group_by_repo:
        lines = [_fmt(r) for r in rows]
        header = f"{len(lines)} tasks"
        if repo:
            header += f" (repo={repo})"
        if worktree_current and wt_filter:
            header += f" (worktree={wt_filter})"
        return f"{header}:\n" + "\n".join(lines)

    # Grouped output — sort repos with named groups first, then unassigned.
    groups: dict[str, list] = {}
    for r in rows:
        try:
            key = r["repo"] or ""
        except (IndexError, KeyError):
            key = ""
        groups.setdefault(key, []).append(r)
    named = sorted(k for k in groups if k)
    ordered = named + ([""] if "" in groups else [])
    out_lines: list[str] = [f"{len(rows)} tasks across {len(ordered)} repo(s):"]
    for key in ordered:
        label = key or "(unassigned)"
        out_lines.append(f"\n## {label} ({len(groups[key])})")
        out_lines.extend(_fmt(r) for r in groups[key])
    return "\n".join(out_lines)


@mcp.tool()
@requires_capability("none")
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
    from .plan_executor import PlanValidationError, TIER_MAP, validate_plan_file

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
@requires_capability("embedding")
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

    # Degraded-search (M1 #8): when no embedding backend is available, fall
    # back to FTS5 BM25 across tasks/skills/plugins so the user still gets
    # ranked keyword hits instead of an error.
    if not embed_available():
        cats = {c.strip() for c in categories.split(",")}
        show_all = "all" in cats
        parts: list[str] = [
            "<!-- search_context mode=keyword-fts5 "
            "(degraded-search: embeddings unavailable) -->",
        ]

        if show_all or "tasks" in cats:
            task_hits = _store.search_text(query, tables=["tasks"],
                                           top_k=top_k, status="open")
            if task_hits:
                task_lines = [
                    f"### Task #{t['id']}: {t['title_or_rule']} (bm25={t['score']:.2f})\n"
                    f"{t['summary_or_why']}"
                    for t in task_hits
                ]
                parts.append("## Open Tasks (keyword)\n\n" + "\n\n".join(task_lines))

        if show_all or "closed" in cats:
            closed_hits = _store.search_text(query, tables=["tasks"],
                                              top_k=3, status="closed")
            if closed_hits:
                closed_lines = [
                    f"### Closed #{t['id']}: {t['title_or_rule']} (bm25={t['score']:.2f})\n"
                    f"{t['summary_or_why'] or ''}"[:400]
                    for t in closed_hits
                ]
                parts.append("## Related Past Work (keyword)\n\n" + "\n\n".join(closed_lines))

        if show_all or "skills" in cats:
            skill_hits = _store.search_skills_text(query, top_k=top_k)
            if skill_hits:
                skill_lines = [
                    f"- {s['id']}: {(s['description'] or '')[:100]}"
                    for s in skill_hits
                ]
                parts.append("## Matching Skills (keyword)\n\n" + "\n".join(skill_lines))
                _last_search_state["skills"] = [s["id"] for s in skill_hits]

        if show_all or "plugins" in cats:
            plugin_hits = _store.suggest_plugins_text(query, top_k=top_k)
            if plugin_hits:
                enabled_plugins: dict = {}
                if SETTINGS_PATH.exists():
                    settings = json.loads(SETTINGS_PATH.read_text())
                    enabled_plugins = settings.get("enabledPlugins", {})
                disabled = [s for s in plugin_hits[:3]
                            if not enabled_plugins.get(s["plugin_id"], False)]
                if disabled:
                    plug_lines = [
                        f"- {s['short_name']}: {(s['description'] or '')[:80]}"
                        for s in disabled
                    ]
                    parts.append(
                        "## Disabled Plugins That May Help (keyword)\n\n"
                        + "\n".join(plug_lines)
                        + "\nUse toggle_plugin() to enable."
                    )

        _last_search_state["query"] = query
        _last_search_state["vector"] = []

        if len(parts) == 1:
            return (
                parts[0] + "\n\n"
                "No relevant context found via keyword fallback. "
                "Enable an embedding backend (Ollama or "
                "sentence-transformers) for semantic search."
            )
        return "\n\n---\n\n".join(parts)

    query_vector = embed(query)

    if not _session["topic"]:
        _session["topic"] = query
        _session["topic_vector"] = query_vector

    _last_search_state["query"] = query
    _last_search_state["vector"] = query_vector

    cats = {c.strip() for c in categories.split(",")}
    show_all = "all" in cats

    parts: list[str] = ["<!-- search_context mode=vector -->"]

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

    # 5. Wiki knowledge layer — explicit search with private-access gate.
    # ``wiki`` is always searched; ``wiki-private`` only for authorized scopes.
    # Index hits (slug == "index" or rel_path contains "index.md") are promoted
    # to the top so the curated overview ranks above raw log/source pages.
    try:
        from . import wiki as _wiki_mod
        from pathlib import Path as _wpath
        _wiki_root = _wpath(_cfg.get("wiki_root") or
                            _wpath.home() / ".claude" / "mcp-skill-hub" / "wiki")
        _wiki_auth = _wiki_authorized_scopes()
        _wiki_ns = ["wiki"] + (["wiki-private"] if _wiki_auth else [])
        _wiki_hits = _store.search_vectors(
            query, namespaces=_wiki_ns, top_k=top_k * 2
        )
        if _wiki_hits:
            # Promote index.md / slug=="index" hits to front.
            def _is_index_hit(r: dict) -> bool:
                meta = r.get("metadata") or {}
                if isinstance(meta, str):
                    try:
                        import json as _json; meta = _json.loads(meta)
                    except Exception:  # noqa: BLE001
                        meta = {}
                rel = str(meta.get("rel_path") or "")
                slug = str(meta.get("slug") or "")
                return "index" in slug or rel.endswith("index.md")
            _wiki_hits = sorted(
                _wiki_hits, key=lambda r: (0 if _is_index_hit(r) else 1, -r.get("score", 0))
            )
            wiki_lines = []
            for r in _wiki_hits[:top_k]:
                meta = r.get("metadata") or {}
                if isinstance(meta, str):
                    try:
                        import json as _json2; meta = _json2.loads(meta)
                    except Exception:  # noqa: BLE001
                        meta = {}
                ns = r.get("namespace", "")
                # Gate: private results only for authorized scopes.
                if ns == "wiki-private":
                    if not _wiki_auth:
                        continue
                slug = meta.get("slug") or r.get("doc_id", "")
                title = meta.get("title") or slug
                rel = meta.get("rel_path") or ""
                wiki_lines.append(
                    f"- [[{slug}]] — {title} "
                    f"(score={r.get('score', 0):.2f}, {ns}, {rel})"
                )
            if wiki_lines:
                parts.append("## Wiki Knowledge\n\n" + "\n".join(wiki_lines))
    except Exception:  # noqa: BLE001
        pass

    # 6. Plugin memory (A4 + M2) — vectors from plugin-declared indexes.
    # Includes both the legacy ``memory:<plugin>`` namespace AND any custom
    # indexes a plugin declares via plugin.json ``vector_indexes`` (e.g.
    # ``career:profile``, ``career:narrative``).
    # Wiki namespaces are excluded here — they are surfaced in section 5 above.
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
                and str(r.get("namespace", "")) not in ("wiki", "wiki-private")
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

    # 7. User identity + habits — surface only when relevant to the query.
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

    # 8. CodeGraph symbols — injected when the flag is on and an index exists.
    if _cfg.get("search_context_use_codegraph"):
        try:
            from .codegraph_context import get_context_block, has_codegraph_index
            from pathlib import Path as _Path
            _repo_root = _Path.cwd()
            if has_codegraph_index(_repo_root):
                cg_block = get_context_block(query, _repo_root)
                if cg_block:
                    parts.append(cg_block)
        except Exception:  # noqa: BLE001
            pass

    # parts[0] is the mode marker added at the start; len==1 means no real hits.
    if len(parts) <= 1:
        return "No relevant context found. Try index_skills() and index_plugins() first."

    return maybe_compress(
        "\n\n---\n\n".join(parts), context=query, site="search_context"
    )


# ---------------------------------------------------------------------------
# Management


@mcp.tool()
@requires_capability("none")
def index_skills() -> str:
    """Rebuild the skill index from all plugin directories."""
    log_tool("index_skills")
    if not embed_available():
        return embed_unavailable_reason()

    count, errors = index_all(_store)
    result = f"Indexed {count} skills."
    if errors:
        result += f"\n\nErrors ({len(errors)}):\n" + "\n".join(f"  - {e}" for e in errors[:10])
    return result


@mcp.tool()
@requires_capability("embedding")
def index_plugins() -> str:
    """Index plugin descriptions for suggest_plugins()."""
    log_tool("index_plugins")
    if not embed_available():
        return embed_unavailable_reason()

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
        import re
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
@requires_capability("none")
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
@requires_capability("none")
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
@requires_capability("none")
def toggle_plugin(plugin_name: str, enabled: bool) -> str:
    """Enable or disable a plugin. Takes effect on next session restart."""
    from .plugin_registry import toggle as _toggle

    return _toggle(plugin_name, enabled)


# ──────────────────────────────────────────────────────────────────────
# S3 F-SELECT — profile-based plugin curation
# ──────────────────────────────────────────────────────────────────────


@mcp.tool()
@requires_capability("none")
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
@requires_capability("none")
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
@requires_capability("none")
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
@requires_capability("none")
def delete_profile(name: str) -> str:
    """Delete a saved profile by name."""
    from . import profiles as _prof

    return "deleted" if _prof.delete_profile(_store, name) else f"no such profile: {name}"


# ──────────────────────────────────────────────────────────────────────
# S4 F-ROUTE — ε-greedy bandit over model tiers
# ──────────────────────────────────────────────────────────────────────


@mcp.tool()
@requires_capability("none")
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
@requires_capability("none")
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
@requires_capability("none")
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
@requires_capability("none")
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
@requires_capability("none")
def list_prompt_rewriters() -> str:
    """List registered prompt rewriters (S5 F-PROMPT)."""
    from .router import rewriters as _rw

    return "rewriters:\n" + "\n".join(f"  - {n}" for n in _rw.available())


@mcp.tool()
@requires_capability("none")
def team_plan(task_kind: str, effort: str = "xhigh", estimate: bool = False) -> str:
    """Return a specialized team orchestration plan for the given task kind.

    ``task_kind`` must be one of: review, arch, issues, implement.
    ``effort`` controls the model floor and verification loops:
    low | medium | high | xhigh (default).
    ``estimate`` appends a heuristic cost projection when True.
    """
    from .team import policy

    try:
        plan = policy.resolve_team_plan(task_kind, effort)
    except ValueError as exc:
        return str(exc)

    lines: list[str] = [
        f"task_kind : {plan['task_kind']}",
        f"effort    : {plan['effort']}",
        f"substrate : {plan['substrate']}",
        f"loops     : {plan['loops']}",
        "",
        "roster:",
    ]
    for entry in plan["roles"]:
        role_label = entry["role"]
        if "lens" in entry:
            role_label = f"{entry['role']}({entry['lens']})"
        lines.append(
            f"  {role_label:<36} -> {entry['agent']:<32}  [{entry['cc_model']} / {entry['tier']}]"
        )

    if estimate:
        try:
            est = policy.estimate_cost(task_kind, effort)
        except ValueError as exc:
            return "\n".join(lines) + f"\n\nestimate error: {exc}"

        lines += [
            "",
            "estimate (heuristic ±50%):",
            f"  agent_calls       : {est['agent_calls']}",
            f"  token_budget_low  : {est['token_budget_low']:,}",
            f"  token_budget_high : {est['token_budget_high']:,}",
            f"  rough_minutes_low : {est['rough_minutes_low']}",
            f"  rough_minutes_high: {est['rough_minutes_high']}",
            "  assumptions:",
        ]
        for assumption in est["assumptions"]:
            lines.append(f"    - {assumption}")

    return "\n".join(lines)


@mcp.tool()
@requires_capability("none")
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
@requires_capability("none")
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
@requires_capability("none")
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
@requires_capability("llm")
def optimize_memory(dry_run: bool = True, bypass_gate: bool = False) -> str:
    """Analyze memory files with tier-smart LLM routing. Recommends KEEP/PRUNE/COMPACT/MERGE. dry_run=True for report only.

    ``bypass_gate=True`` skips the internal IDLE-only pressure check. It is
    intended ONLY for callers that have already cleared their own (more
    permissive) gate — e.g. the postcompact path, which runs up to LOW
    pressure. Do not set it for background/nightly callers.
    """
    log_tool("optimize_memory", dry_run=dry_run)

    if not bypass_gate and not should_run_llm("optimize_memory"):
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

    # Deterministic-first: shrink each file losslessly before spending LLM
    # tokens on it. compress_payload() runs the dependency-free JSON-minify +
    # duplicate-line collapse pass (and the headroom router when installed);
    # it passes prose through untouched, so this only helps and never garbles
    # the content the classifier reads. Lets more files fit per pass.
    from .compression import compress_payload
    det_saved = 0
    for e in entries:
        try:
            payload = compress_payload(e["content"], allow_lossy=False)
        except Exception:  # noqa: BLE001 — compression must never break analysis
            continue
        if not payload.lossy and payload.bytes_after < payload.bytes_before:
            det_saved += payload.bytes_before - payload.bytes_after
            e["content"] = payload.compressed
    if det_saved:
        lines.append(
            f"Deterministic pre-compression saved ~{det_saved // 4:,} tokens "
            f"(lossless, no LLM) before classification.\n"
        )

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
            max_tokens=int(_cfg.get("optimize_memory_max_tokens") or 4000),
            temperature=0.0,
            timeout=300.0,
            op="optimize_context",
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
@requires_capability("none")
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

        from . import capabilities as _cap
        if _cap.no_llm_mode_active():
            ns = _cap.no_llm_summary()
            lines.append(
                f"No-LLM mode:     ON ({ns['available']}/{ns['total']} tools available)"
            )
            lines.append(
                f"                 {ns['disabled']} tool(s) disabled — see /status/capabilities"
            )
        else:
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
@requires_capability("none")
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


def _compression_report() -> str:
    """Build the compression-savings section for token_stats (always shown)."""
    try:
        c = _store.get_compression_stats()
    except Exception:  # noqa: BLE001
        return ""
    from skill_hub.compression import is_available as _compression_available

    headroom_installed = _compression_available()
    master = "on" if _cfg.get("compression_enabled") else "off"
    # Show whether the lossy Kompress path is actually reachable at runtime:
    # config says "on" but headroom-ai absent → flag the no-op explicitly.
    if _cfg.get("compression_ml_enabled"):
        ml = "on" if headroom_installed else "on (headroom-ai not installed — no-op; install compression_full extra)"
    else:
        ml = "off"
    code = "on" if _cfg.get("compression_code_aware_enabled") else "off"
    headroom_line = "headroom-ai=installed" if headroom_installed else "headroom-ai=missing (deterministic built-ins only)"
    if not c.get("calls"):
        return (
            "\n=== Tool-output Compression (headroom) ===\n"
            f"  master={master}  ml/Kompress={ml}  code-aware={code}\n"
            f"  {headroom_line}\n"
            "  No compression activity recorded yet."
        )
    saved = c["saved"]
    tok = c["tokens_saved"]
    hit_rate = (c["hits"] / c["calls"] * 100) if c["calls"] else 0.0
    lines = [
        "\n=== Tool-output Compression (headroom) ===",
        f"  master={master}  ml/Kompress={ml}  code-aware={code}",
        f"  {headroom_line}",
        f"  Calls: {c['calls']:,}  ({c['hits']:,} compressed, {hit_rate:.0f}% hit-rate)",
        f"  Bytes: {c['bytes_before']:,} → {c['bytes_after']:,}  (saved {saved:,} bytes)",
        f"  Tokens saved (est.): ~{tok:,}"
        f"  (~${tok / 1_000_000 * 3:.4f} at $3/M)",
        f"  Avg ratio (over hits): {c['avg_ratio']:.3f}  (lower = better)",
    ]
    by_strat = c.get("by_strategy") or {}
    if by_strat:
        lines.append("  By strategy:")
        for name, d in sorted(by_strat.items(), key=lambda kv: -kv[1]["saved"]):
            lines.append(
                f"    {name:<14} {d['count']:>4}x  ~{d['saved']:,} bytes saved"
            )
    return "\n".join(lines)


def _llm_report() -> str:
    """Build the local-LLM metering section for token_stats (always shown)."""
    try:
        s = _store.get_llm_stats()
    except Exception:  # noqa: BLE001
        return ""
    metering = "on" if _cfg.get("llm_metering_enabled") else "off"
    if not s.get("calls"):
        return (
            "\n=== Local LLM Calls (Ollama) ===\n"
            f"  metering={metering}\n"
            "  No local LLM calls recorded yet."
        )
    lines = [
        "\n=== Local LLM Calls (Ollama) ===",
        f"  metering={metering}",
        f"  Calls: {s['calls']}  ({s['errors']} errors)",
        f"  Latency: avg ~{s['avg_latency_ms']:.0f} ms",
        f"  Tokens: prompt {s['prompt_tokens']}  completion {s['completion_tokens']}"
        f"  (total {s['total_tokens']})",
        f"  Throughput: ~{s['tokens_per_sec']:.1f} tok/s",
    ]
    by_op = s.get("by_op") or {}
    if by_op:
        lines.append("  By op:")
        for op_name, d in sorted(by_op.items(), key=lambda kv: -kv[1]["count"]):
            lines.append(
                f"    {op_name:<20} {d['count']}x  ~{d['total_tokens']:,} tokens"
                f"  (~{d['duration_ms']:,}ms)"
            )
    return "\n".join(lines)


@mcp.tool()
@requires_capability("none")
def token_stats() -> str:
    """Show estimated token savings from hook interceptions and tool-output compression."""
    compression = _compression_report()
    totals = _store.get_interception_totals()
    if not totals or not totals["total_interceptions"]:
        enabled = _cfg.get("token_profiling")
        if not enabled:
            head = (
                "Token profiling is disabled.\n"
                "Enable with: configure(key='token_profiling', value='true')"
            )
        else:
            head = "No interceptions recorded yet. Use task commands to build up data."
        llm = _llm_report()
        suffix = ""
        if compression:
            suffix += "\n" + compression
        if llm:
            suffix += "\n" + llm
        return head + suffix

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
    if compression:
        lines.append(compression)
    llm = _llm_report()
    if llm:
        lines.append(llm)
    return "\n".join(lines)


@mcp.tool()
@requires_capability("none")
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
@requires_capability("none")
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
@requires_capability("none")
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
@requires_capability("none")
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
@requires_capability("none")
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
@requires_capability("none")
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
    "lint_canary": {
        "cron": "15 4 * * *",  # Daily 4:15 AM
        "description": (
            "Daily lint-canary: rotates through ruff selectors "
            "(F841/F821/B023/S701/RUF034/RUF006/B026/...) and records findings."
        ),
        "enabled_default": False,
        "prompt": (
            "Run lint_canary(). Report which selector was checked, how many "
            "findings were captured, and whether the rotation cursor advanced."
        ),
    },
}


@mcp.tool()
@requires_capability("none")
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
@requires_capability("none")
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
@requires_capability("none")
def lint_canary(target: str = ".", selectors: list[str] | None = None) -> str:
    """M3 #17 — Run one step of the lint-canary cadence.

    Picks the next ruff selector from the rotation list (F841/F821/B023/...),
    runs ``ruff check --select <selector> <target>``, advances the cursor, and
    appends a JSONL record to the witness log under
    ``~/.claude/mcp-skill-hub/state/witness_log.jsonl``.

    Pass ``selectors`` to override (and persist) the rotation list.
    """
    from .lint_canary import run_lint_canary, format_run

    run = run_lint_canary(target=target, selectors=selectors)
    log_tool(
        "lint_canary",
        selector=run.selector,
        findings=run.findings,
        cursor=run.cursor_after,
    )
    return format_run(run)


@mcp.tool()
@requires_capability("none")
def record_witness(
    issue: str,
    pr: str,
    sha: str,
    repo: str,
    fix_kind: str = "fix",
    fix_summary: str = "",
) -> str:
    """M1 #10 — Append a fix-manifest entry to the witness log.

    Records ``(issue, pr, sha, repo, fix_kind, fix_summary)`` as one JSONL
    line under ``~/.claude/mcp-skill-hub/state/witness_log.jsonl``. The log is
    append-only; existing entries are never edited or removed by this tool.

    Use this after merging a fix so the dashboard can show a real fix history
    (vs. relying on memory files). ``fix_kind`` is the conventional-commit
    label ("feat" / "fix" / "refactor" / etc.).
    """
    from .witness import record_witness as _record

    try:
        rec = _record(
            issue=issue,
            pr=pr,
            sha=sha,
            repo=repo,
            fix_kind=fix_kind,
            fix_summary=fix_summary,
        )
    except ValueError as exc:
        return f"record_witness failed: {exc}"
    except OSError as exc:
        return f"record_witness failed: cannot write log ({exc})"
    log_tool(
        "record_witness",
        repo=rec.repo,
        issue=rec.issue,
        pr=rec.pr,
        sha=rec.sha,
        fix_kind=rec.fix_kind,
    )
    return (
        f"witness recorded: {rec.repo} {rec.fix_kind} {rec.issue} "
        f"pr={rec.pr} sha={rec.sha}"
    )


@mcp.tool()
@requires_capability("none")
def list_witness(
    repo: str = "",
    since: int = 0,
    limit: int = 0,
) -> str:
    """M1 #10 — List recorded fix-manifest entries, newest first.

    Parameters
    ----------
    repo:  Optional ``"owner/name"`` filter (exact match). Empty -> all repos.
    since: Optional minimum epoch-seconds (inclusive). 0 -> all timestamps.
    limit: Optional max records to return. 0 -> no limit.
    """
    from .witness import list_witness as _list, format_witness_list

    records = _list(
        repo=(repo or None),
        since=(since or None),
        limit=(limit or None),
    )
    log_tool(
        "list_witness",
        repo=repo,
        since=since,
        limit=limit,
        matches=len(records),
    )
    return format_witness_list(records)


@mcp.tool()
@requires_capability("none")
def sync_check(
    primary: str,
    followers: list[str],
    base_ref: str = "HEAD~1",
    removed_symbols: list[str] | None = None,
) -> str:
    """M3 #16 — Cross-repo stale-import detector.

    Greps follower repos for symbols recently removed/renamed in the primary
    (SSOT) repo. Reports lines like
    ``stale ref "OldClass" in follower/src/foo.py:42``.

    Pure grep — no LLM. The primary diff is computed via
    ``git diff <base_ref>``; symbols that still exist anywhere in the primary
    working tree are filtered out (so renames/moves don't generate false
    positives). Pass ``removed_symbols`` to bypass the diff step and grep
    for a caller-supplied list.
    """
    from .sync_check import sync_check as _sc, format_report

    report = _sc(
        primary=primary,
        followers=followers,
        base_ref=base_ref,
        removed_symbols=removed_symbols,
    )
    log_tool(
        "sync_check",
        primary=primary,
        base_ref=base_ref,
        followers=len(followers),
        removed_symbols=len(report.removed_symbols),
        findings=len(report.findings),
    )
    return format_report(report)


@mcp.tool()
@requires_capability("none")
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
@requires_capability("llm")
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
@requires_capability("none")
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
@requires_capability("llm")
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
    # Wiki namespaces are derived from markdown (source of truth) and are
    # rebuilt by wiki_reindex — never promoted or pruned here.
    _wiki_guard = "namespace NOT IN ('wiki','wiki-private')"
    rules = [
        ("promote", "L1", "L2",
         f"level = 'L1' AND access_count >= 2 AND "
         f"indexed_at < datetime('now', '-7 days') AND {_wiki_guard}"),
        ("promote", "L2", "L3",
         f"level = 'L2' AND access_count >= 5 AND "
         f"indexed_at < datetime('now', '-30 days') AND {_wiki_guard}"),
        ("prune",   "L0", None,
         f"level = 'L0' AND access_count = 0 AND "
         f"indexed_at < datetime('now', '-1 day') AND {_wiki_guard}"),
        ("prune",   "L1", None,
         f"level = 'L1' AND access_count = 0 AND "
         f"indexed_at < datetime('now', '-7 days') AND {_wiki_guard}"),
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
@requires_capability("embedding")
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
@requires_capability("none")
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

    return maybe_compress("\n".join(lines), context=query, site="search_web")


@mcp.tool()
@requires_capability("none")
def analyze_router_log(
    top_n: int = 20,
    propose_teachings: bool = True,
    window_size: int = 10,
) -> str:
    """Analyse the prompt-router audit log to surface misclassifications and suggest teach() rules.

    Reads router.jsonl, clusters verdicts by tier and model, identifies patterns
    where confidence was low or where the user might have disagreed, and
    optionally proposes teach() calls to improve future routing.

    Also reports a sliding-window confidence trend (recent vs prior window) and
    flags expensive-tier verdicts that may be candidates for downgrade.

    Args:
        top_n: Number of recent entries to analyse (default 20).
        propose_teachings: If True, include suggested teach() calls (default True).
        window_size: Size of each window for trend analysis (default 10).
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
    # Field names from verdict.py append_audit_log (authoritative schema):
    #   verdict.tier   (not tier_used)
    #   latency.total_ms (not latency_ms)
    #   prompt         (not prompt_preview)
    model_counts: Counter = Counter(
        e.get("verdict", {}).get("model", "unknown") for e in entries
    )
    tier_counts: Counter = Counter(
        e.get("verdict", {}).get("tier", "?") for e in entries
    )
    plan_count = sum(1 for e in entries if e.get("verdict", {}).get("plan_mode"))
    confs = [e.get("verdict", {}).get("confidence", 0.0) for e in entries]
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    avg_lat = (
        sum(e.get("latency", {}).get("total_ms", 0) for e in entries) / len(entries)
    )

    lines_out: list[str] = [
        f"## Router Log Analysis — last {len(entries)} prompts\n",
        f"Model distribution: {dict(model_counts)}",
        f"Tier distribution:  {dict(tier_counts)}",
        f"Plan mode triggered: {plan_count}/{len(entries)}",
        f"Avg confidence: {avg_conf:.2f}  |  Avg latency: {int(avg_lat)}ms\n",
    ]

    # ── Sliding-window trend ──────────────────────────────────────────────────
    w = max(1, window_size)
    if len(entries) >= 2:
        recent = entries[-w:]
        prior = entries[-(2 * w):-w] if len(entries) >= 2 * w else entries[:-w]
        def _window_stats(window: list[dict]) -> dict:
            if not window:
                return {"avg_conf": 0.0, "tiers": {}}
            wc = [e.get("verdict", {}).get("confidence", 0.0) for e in window]
            tc: Counter = Counter(e.get("verdict", {}).get("tier", "?") for e in window)
            return {"avg_conf": sum(wc) / len(wc), "tiers": dict(tc)}
        rs = _window_stats(recent)
        ps = _window_stats(prior) if prior else None
        trend_lines = [f"### Sliding-window trend (window={w})"]
        trend_lines.append(
            f"  Recent  ({len(recent)}): avg_conf={rs['avg_conf']:.2f}  tiers={rs['tiers']}"
        )
        if ps:
            delta = rs["avg_conf"] - ps["avg_conf"]
            direction = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "→")
            trend_lines.append(
                f"  Prior   ({len(prior)}): avg_conf={ps['avg_conf']:.2f}  tiers={ps['tiers']}"
            )
            trend_lines.append(f"  Confidence delta: {delta:+.2f} {direction}")
        else:
            trend_lines.append("  (not enough entries for a prior window)")
        lines_out.append("\n".join(trend_lines))

    # ── Cost-benefit: expensive-tier + high-conf + low-complexity ────────────
    from . import model_registry as _mr
    downgrade_candidates: list[str] = []
    total_usd_saved = 0.0
    for e in entries:
        v = e.get("verdict", {})
        savings = e.get("savings", {})
        total_usd_saved += savings.get("usd_saved", 0.0)
        tier = v.get("tier", 0)
        confidence = v.get("confidence", 0.0)
        complexity = v.get("complexity", 1.0)
        model = v.get("model", "")
        blended = _mr.blended_usd_per_m(model)
        # Flag: used an expensive model (blended >= haiku*2 i.e. >= 2.0) at
        # high confidence (>= 0.80) on low complexity (< 0.40).
        if (
            blended is not None
            and blended >= 2.0
            and confidence >= 0.80
            and complexity < 0.40
        ):
            prompt = e.get("prompt", "")
            downgrade_candidates.append(
                f"  tier={tier} model={model} (${blended:.1f}/M) "
                f"conf={confidence:.2f} complexity={complexity:.2f}  "
                f"\"{prompt[:60]}\""
            )

    lines_out.append(
        f"\n### Cost summary\n"
        f"  Total usd_saved (logged): ${total_usd_saved:.4f}"
    )
    if downgrade_candidates:
        lines_out.append(
            f"\n### Downgrade opportunities "
            f"(high-conf + low-complexity on expensive tier): "
            f"{len(downgrade_candidates)}"
        )
        for line in downgrade_candidates[:5]:
            lines_out.append(line)
        lines_out.append(
            "  → These verdicts used an expensive model on a simple, "
            "high-confidence prompt. Consider lowering the complexity threshold "
            "for haiku/cheaper routing."
        )

    # ── Low-confidence entries ────────────────────────────────────────────────
    low_conf = [e for e in entries if e.get("verdict", {}).get("confidence", 1.0) < 0.65]
    if low_conf:
        lines_out.append(f"\n### Low-confidence verdicts ({len(low_conf)})")
        for e in low_conf[:5]:
            v = e.get("verdict", {})
            lines_out.append(
                f"  - [tier={v.get('tier', '?')}] conf={v.get('confidence', 0.0):.2f} "
                f"→ {v.get('model', '?')}  \"{e.get('prompt', '')}\""
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
            preview = e.get("prompt", "")
            if not preview or preview in seen:
                continue
            seen.add(preview)
            v = e.get("verdict", {})
            model_val = v.get("model", "sonnet")
            lines_out.append(
                f'teach(\n'
                f'  rule="when the prompt resembles: {preview[:60]}",\n'
                f'  suggest="model:{model_val}"\n'
                f')'
            )

    # ── Tier-3 batch task usage ───────────────────────────────────────────────
    tier3 = [e for e in entries if e.get("verdict", {}).get("tier") == 3]
    if tier3:
        lines_out.append(f"\n### Tier-3 (Haiku) calls: {len(tier3)}/{len(entries)}")
        lines_out.append("Consider raising router_haiku_threshold if Haiku fires too often.")

    return "\n".join(lines_out)


@mcp.tool()
@requires_capability("none")
def analyze_failures(
    hours: int = 24,
    min_count: int = 2,
) -> str:
    """Cluster recurring tool failures from the event log to surface patterns.

    Scans ``tool_result`` events with ``ok=False`` from the last *hours* hours,
    normalises error strings, groups by (tool_name, normalised_error), and
    returns a Markdown table of clusters with count >= *min_count* plus a
    drafted (not filed) issue-body block for each top cluster.

    No LLM is used — purely deterministic DB read.
    """
    log_tool("analyze_failures", hours=hours, min_count=min_count)
    from .log_insights import cluster_failures
    result = cluster_failures(hours=hours, min_count=min_count)
    clusters = result.get("clusters", [])
    scanned = result.get("scanned", 0)
    if not clusters:
        return (
            f"analyze_failures: no recurring failures found "
            f"(scanned {scanned} tool_result events in the last {hours}h, "
            f"min_count={min_count})."
        )
    # Markdown table
    lines: list[str] = [
        f"## Recurring tool failures — last {hours}h\n",
        f"Scanned {scanned} tool_result events.  "
        f"Clusters with count >= {min_count}: **{len(clusters)}**\n",
        "| # | tool | pattern | count | first_ts | last_ts |",
        "|---|------|---------|-------|----------|---------|",
    ]
    for i, c in enumerate(clusters, 1):
        lines.append(
            f"| {i} | `{c['tool']}` | {c['pattern'][:60]} "
            f"| {c['count']} | {c['first_ts'][:16]} | {c['last_ts'][:16]} |"
        )
    # Draft issue bodies for top-3 clusters (NOT filed)
    lines.append("\n---\n### Drafted issue bodies (not filed — review before opening)\n")
    for c in clusters[:3]:
        lines.append(
            f"**Issue draft: {c['tool']} — {c['pattern'][:50]}**\n"
            f"```\n"
            f"Title: Recurring failure in `{c['tool']}`: {c['pattern'][:60]}\n\n"
            f"Occurrences: {c['count']} in last {hours}h "
            f"(first: {c['first_ts'][:19]}, last: {c['last_ts'][:19]})\n\n"
            f"Example error:\n{c['example'][:300]}\n\n"
            f"Suggested action: {c['suggested_action']}\n"
            f"```\n"
        )
    return "\n".join(lines)


@mcp.tool()
@requires_capability("none")
def analyze_skill_selection(
    limit: int = 500,
) -> str:
    """Report per-skill injection counts, used counts, used-rate, and feedback.

    Uses ``skill_injections``, ``events`` (kind='skill.used'), and
    ``feedback`` tables — no LLM.

    ``used_rate`` = used / injections per skill.  Skills with injections > 0
    and used == 0 are flagged "injected-but-unused" — real candidates for
    removal or rewrite, derived from actual tool-invocation data rather than
    heuristics.
    """
    log_tool("analyze_skill_selection", limit=limit)
    from .log_insights import skill_selection_stats
    stats = skill_selection_stats(limit=limit)
    rows = stats.get("skills", [])
    total_injections = stats.get("total_injections", 0)
    total_used = stats.get("total_used", 0)
    total_feedback = stats.get("total_feedback", 0)
    if not rows:
        return (
            "analyze_skill_selection: no injection data found. "
            "Run search_skills at least once to populate skill_injections."
        )
    lines: list[str] = [
        f"## Skill selection metrics (last {limit} injections)\n",
        f"Total injections: {total_injections}  |  "
        f"Total used: {total_used}  |  Total feedback rows: {total_feedback}\n",
        "| skill | injections | used | used_rate | helpful_rate | feedback_n | status |",
        "|-------|-----------|------|-----------|-------------|------------|--------|",
    ]
    for r in rows:
        helpful_rate = r.get("helpful_rate")
        rate_str = f"{helpful_rate:.0%}" if helpful_rate is not None else "n/a"
        used_rate = r.get("used_rate")
        used_rate_str = f"{used_rate:.0%}" if used_rate is not None else "n/a"
        status = r.get("status", "")
        lines.append(
            f"| `{r['skill_id']}` | {r['injections']} | {r.get('used', 0)} "
            f"| {used_rate_str} | {rate_str} | {r['feedback_n']} | {status} |"
        )
    # Surface injected-but-unused candidates explicitly
    unused = [r for r in rows if r.get("status") == "injected-but-unused"]
    if unused:
        lines.append(
            f"\n**Injected-but-unused candidates ({len(unused)})**: "
            + ", ".join(f"`{r['skill_id']}`" for r in unused)
        )
    return "\n".join(lines)


@mcp.tool()
@requires_capability("none")
def log_digest(hours: int = 24) -> str:
    """Return a markdown digest of recent activity — deterministic, no LLM.

    Sections: Activity summary / Recurring failures / Skill selection.
    Data is read from the structured event log and skill_injections tables.
    """
    log_tool("log_digest", hours=hours)
    from .log_insights import build_digest
    d = build_digest(hours=hours)

    lines: list[str] = [
        f"## Log Digest — {d['window']}\n",
        "### Activity",
        f"- Events: **{d['total']}** across **{d['distinct_sessions']}** session(s)",
    ]
    if d["by_kind"]:
        lines.append("\n| kind | count |")
        lines.append("|------|-------|")
        for kind, count in sorted(d["by_kind"].items(), key=lambda kv: -kv[1]):
            lines.append(f"| {kind} | {count} |")
    else:
        lines.append("\n_No events in this window._")

    lines.append("\n### Recurring Failures")
    if d["top_failures"]:
        lines.append("| tool | pattern | count | last seen |")
        lines.append("|------|---------|-------|-----------|")
        for c in d["top_failures"]:
            lines.append(
                f"| `{c['tool']}` | {c['pattern'][:60]} "
                f"| {c['count']} | {c['last_ts'][:16]} |"
            )
    else:
        lines.append("_No recurring failures (threshold: min 2 occurrences)._")

    lines.append("\n### Skill Selection")
    lines.append(
        f"Total injections: {d['total_injections']}  |  "
        f"Total feedback rows: {d['total_feedback']}"
    )
    if d["skills"]:
        lines.append("\n| skill | injections | helpful_rate | status |")
        lines.append("|-------|-----------|-------------|--------|")
        for s in d["skills"]:
            rate = s.get("helpful_rate")
            rate_str = f"{rate:.0%}" if rate is not None else "n/a"
            lines.append(
                f"| `{s['skill_id']}` | {s['injections']} "
                f"| {rate_str} | {s.get('status', '')} |"
            )
    else:
        lines.append("\n_No injection data yet._")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session memory (ported from cookbook session_memory_compaction.ipynb)

@mcp.tool()
@requires_capability("none")
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
@requires_capability("none")
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


# ---------------------------------------------------------------------------
# M2 W1 — Event log tools


@mcp.tool()
@requires_capability("none")
def get_events(
    session_id: str = "",
    since: float = 0.0,
    kind: str = "",
    limit: int = 200,
) -> str:
    """Query the M2 W1 event log.

    All parameters are optional; omitting ``session_id`` returns events across
    all sessions (bounded by ``limit``).

    Args:
        session_id: Filter to one session.  Empty = all sessions.
        since:      Unix timestamp lower bound (inclusive).  0 = no bound.
        kind:       Filter by event kind (tool_invoke | tool_result |
                    session_start | session_end | config_change |
                    session_snapshot).  Empty = all kinds.
        limit:      Maximum number of rows to return (capped at 10 000).
    """
    log_tool("get_events", session_id=session_id[:36], kind=kind, limit=limit)
    rows = _store.get_events(session_id=session_id, since=since, kind=kind, limit=limit)
    if not rows:
        return "No events found."
    lines = []
    for r in rows:
        ts_str = f"{r['ts']:.3f}"
        name_str = f" [{r['tool_name']}]" if r.get("tool_name") else ""
        lines.append(
            f"id={r['id']} ts={ts_str} kind={r['kind']}{name_str} "
            f"session={r['session_id'][:12]} payload={r['payload'][:120]}"
        )
    return f"{len(rows)} event(s):\n" + "\n".join(lines)


@mcp.tool()
@requires_capability("none")
def events_prune(before_ts: float = 0.0, dry_run: bool = False) -> str:
    """Prune the event log: coalesce closed-session events older than the retention window.

    Retention is controlled by config ``event_log_retention_days`` (default 30;
    ``0`` = keep forever).  ``before_ts`` overrides the config window when > 0.

    In-flight sessions (no ``session_end`` event) are never pruned.
    Already-coalesced sessions (only ``session_snapshot`` rows) are skipped.

    Args:
        before_ts: Unix timestamp cutoff.  Events older than this are eligible.
                   0 = use the config retention window.
        dry_run:   If True, report candidates without deleting anything.
    """
    log_tool("events_prune", dry_run=dry_run, before_ts=before_ts)
    result = _store.events_prune(before_ts=before_ts, dry_run=dry_run)
    mode = "dry_run" if dry_run else "pruned"
    return (
        f"events_prune [{mode}]: candidates={result['candidates']} "
        f"rows_deleted={result['rows_deleted']} "
        f"snapshots_written={result['snapshots_written']}"
    )


@mcp.tool()
@requires_capability("none")
def wake_session(session_id: str) -> str:
    """Stateless recovery — M2 W2.

    Replays the event log for ``session_id`` to restore in-memory state after a
    server restart or mid-invoke kill.

    Recovery steps
    --------------
    1. Read all events for the session in chronological order.
    2. Replay ``record_model_reward`` tool_invoke events whose tool_result was
       never written (i.e. the server was killed during the call).  These are
       re-applied directly to the ``model_rewards`` table so the bandit state is
       consistent with what the session had accumulated.
    3. Invalidate the in-process vector cache so the next search reloads from
       the persistent ``embeddings`` table — the table itself is always current;
       only the in-memory cache needs eviction.
    4. Detect the most-recent in-flight tool_invoke (any tool) with no matching
       tool_result and report it in the output so the caller can decide whether
       to re-invoke it.

    Out of scope (per design)
    -------------------------
    * Snapshotting / replay-cost optimization — measure first.
    * Cross-session linking.

    Args:
        session_id: The session to recover.  Must be a non-empty string.
    """
    import time as _time

    if not session_id:
        return "error: session_id is required"

    log_tool("wake_session", session_id=session_id[:36])
    t_start = _time.monotonic()

    # ------------------------------------------------------------------
    # Step 1: load events in order
    # ------------------------------------------------------------------
    events = _store.get_events(session_id=session_id, limit=10000)
    if not events:
        return f"wake_session: no events found for session {session_id!r}"

    elapsed_load_ms = int((_time.monotonic() - t_start) * 1000)

    # ------------------------------------------------------------------
    # Step 2: detect in-flight tool_invoke (no matching tool_result)
    # ------------------------------------------------------------------
    # Walk events in order, track last seen invoke id per tool_name.
    # An "in-flight" invoke has a tool_invoke but no subsequent tool_result
    # with the same tool_name (we match positionally — within the ordered
    # stream, each invoke is expected to be followed by its result before the
    # next invoke of the same tool, but we only track the most-recent unpaired
    # one across the entire session).
    invoke_stack: dict[str, dict] = {}   # tool_name -> last unmatched invoke event
    replay_targets: list[dict] = []      # record_model_reward invokes with no result

    for ev in events:
        kind = ev.get("kind", "")
        tool_name = ev.get("tool_name") or ""
        if kind == "tool_invoke" and tool_name:
            invoke_stack[tool_name] = ev
        elif kind == "tool_result" and tool_name:
            invoke_stack.pop(tool_name, None)

    # Remaining entries in invoke_stack are in-flight (no tool_result).
    in_flight = list(invoke_stack.values())

    # Identify record_model_reward invokes that never got a result — replay them.
    for ev in in_flight:
        if ev.get("tool_name") == "record_model_reward":
            replay_targets.append(ev)

    # ------------------------------------------------------------------
    # Step 3: replay record_model_reward in-flight invokes
    # ------------------------------------------------------------------
    import json as _json
    from .router import bandit as _bandit

    replayed = 0
    replay_errors: list[str] = []
    for ev in replay_targets:
        try:
            payload = _json.loads(ev.get("payload") or "{}")
            kw = payload.get("kwargs") or payload.get("args") or {}
            if isinstance(kw, dict):
                tier = str(kw.get("tier", ""))
                task_class = str(kw.get("task_class", ""))
                domain = str(kw.get("domain", "_none"))
                success_raw = kw.get("success", "0.5")
                success = float(success_raw)
                if tier and task_class:
                    _bandit.record_reward(_store, tier, task_class, domain, success)
                    replayed += 1
                else:
                    # If task_class was omitted, try to derive it from complexity.
                    complexity_raw = kw.get("complexity", "-1.0")
                    complexity = float(complexity_raw)
                    domain_hints_raw = str(kw.get("domain_hints", ""))
                    hints = [h.strip() for h in domain_hints_raw.split(",") if h.strip()]
                    if tier and complexity >= 0.0:
                        task_class, domain = _bandit.bucket(complexity, hints)
                        _bandit.record_reward(_store, tier, task_class, domain, success)
                        replayed += 1
        except Exception as exc:  # noqa: BLE001
            replay_errors.append(str(exc)[:80])

    # ------------------------------------------------------------------
    # Step 4: invalidate in-process caches
    # ------------------------------------------------------------------
    # Vector cache: force reload from embeddings table on next search.
    _store._vec_cache_valid = False
    _store._vec_cache = {}

    elapsed_total_ms = int((_time.monotonic() - t_start) * 1000)

    # ------------------------------------------------------------------
    # Build summary report
    # ------------------------------------------------------------------
    lines = [
        f"wake_session: session={session_id!r}",
        f"  events_replayed={len(events)}  elapsed_ms={elapsed_total_ms}",
        f"  (load_ms={elapsed_load_ms})",
        f"  bandit_rewards_replayed={replayed}",
        f"  vector_cache_invalidated=true",
    ]

    if in_flight:
        lines.append(f"  in_flight_tools={len(in_flight)}:")
        for ev in in_flight:
            tool = ev.get("tool_name", "?")
            ts = ev.get("ts", 0.0)
            eid = ev.get("id", "?")
            lines.append(f"    - tool={tool}  event_id={eid}  ts={ts:.3f}")
    else:
        lines.append("  in_flight_tools=0 (session completed cleanly)")

    if replay_errors:
        lines.append(f"  replay_errors={len(replay_errors)}: {replay_errors[:3]}")

    if elapsed_total_ms > 500:
        lines.append(
            f"  WARNING: replay exceeded 500ms budget ({elapsed_total_ms}ms). "
            "Consider snapshotting for large sessions."
        )

    return "\n".join(lines)


@mcp.tool()
@requires_capability("none")
def issue_sync(repo: str = "", dry_run: bool = False) -> str:
    """Reconcile linked GitHub issues with their skill-hub tasks.

    For each task↔issue link:
    - If the GitHub issue is CLOSED and the local task is OPEN: close the task
      locally (issue wins).
    - If the local task is CLOSED and the GitHub issue is OPEN: optionally write
      back a comment or close the issue, controlled by config
      ``task_issue_writeback`` (default "off" = no GitHub writes).

    Args:
        repo:    Optional GitHub repo filter ("owner/name"). Empty = all repos.
        dry_run: Report what would happen without making any changes.
    """
    from . import issue_sync as _issue_sync
    from . import config as _cfg

    log_tool("issue_sync", repo=repo, dry_run=dry_run)
    writeback = str(_cfg.load_config().get("task_issue_writeback") or "off")

    sid = _session.get("id", "")

    def _emit(kind: str, tool_name: str | None, payload: dict) -> None:
        try:
            _store.append_event(session_id=sid, kind=kind,
                                tool_name=tool_name, payload=payload)
        except Exception:  # noqa: BLE001
            pass

    report = _issue_sync.reconcile(
        _store,
        repo=repo,
        dry_run=dry_run,
        writeback=writeback,
        emit=_emit,
    )

    mode = "dry_run" if dry_run else "live"
    lines = [
        f"issue_sync [{mode}]: checked={report['checked']} "
        f"tasks_closed={report['tasks_closed']} "
        f"issues_commented={report['issues_commented']} "
        f"issues_closed={report['issues_closed']} "
        f"writeback={report['writeback']}",
    ]
    for entry in report["drift"]:
        direction = entry.get("direction", "?")
        tid = entry.get("task_id")
        num = entry.get("issue_number")
        r = entry.get("repo") or ""
        action = entry.get("action") or entry.get("writeback_mode", "")
        ref = f"{r}#{num}" if r else f"#{num}"
        lines.append(f"  drift [{direction}] task={tid} issue={ref} action={action}")
    return "\n".join(lines)


@mcp.tool()
@requires_capability("none")
def discussions_sync(repo: str = "", dry_run: bool = False, first: int = 50) -> str:
    """Land this repo's GitHub Discussions (+ comments) into the wiki as source pages.

    Each discussion becomes one mechanical wiki ``source`` page (no LLM, comments
    folded in). The scan→approve→ingest loop (wiki_scan / wiki_queue_decision /
    wiki_ingest) distills them; query via wiki_query. Supersedes the old raw
    'discussions' vector namespace.

    Args:
        repo:     GitHub repo as "owner/name". Empty = resolve from current directory.
        dry_run:  Report what would be written without writing pages.
        first:    Number of discussions to fetch (max 100).
    """
    from . import discussions_sync as _discussions_sync

    log_tool("discussions_sync", repo=repo, dry_run=dry_run, first=first)

    sid = _session.get("id", "")

    def _emit(kind: str, tool_name: str | None, payload: dict) -> None:
        try:
            _store.append_event(session_id=sid, kind=kind,
                                tool_name=tool_name, payload=payload)
        except Exception:  # noqa: BLE001
            pass

    report = _discussions_sync.sync_discussions(
        _store,
        repo=repo,
        dry_run=dry_run,
        first=first,
        emit=_emit,
    )

    if "error" in report:
        return f"discussions_sync error: {report['error']}"

    mode = "dry_run" if dry_run else "live"
    lines = [
        f"discussions_sync [{mode}]: "
        f"checked={report['checked']} "
        f"indexed={report['indexed']} "
        f"discussions={report['discussions']} "
        f"comments={report['comments']} "
        f"skipped={report['skipped']}",
    ]
    if dry_run:
        lines.append("  (dry_run: no writes made)")
    return "\n".join(lines)


@mcp.tool()
@requires_capability("embedding")
def index_logs(hours: int = 24, limit: int = 1000, dry_run: bool = False) -> str:
    """Embed recent activity-log events into vector memory (namespace 'logs'), searchable via search_context.

    Args:
        hours:    How many hours back to scan (default 24).
        limit:    Maximum number of events to process (default 1000).
        dry_run:  Report what would be indexed without writing to the DB.
    """
    from . import log_insights as _log_insights

    log_tool("index_logs", hours=hours, limit=limit, dry_run=dry_run)

    report = _log_insights.index_recent_logs(hours=hours, limit=limit, dry_run=dry_run)

    mode = "dry_run" if dry_run else "live"
    by_kind = report.get("by_kind") or {}
    lines = [
        f"index_logs [{mode}]: "
        f"scanned={report['scanned']} "
        f"indexed={report['indexed']} "
        f"namespace={report['namespace']}",
    ]
    if by_kind:
        lines.append("  by_kind:")
        for kind, count in sorted(by_kind.items()):
            lines.append(f"    {kind}: {count}")
    if dry_run:
        lines.append("  (dry_run: no writes made)")
    return "\n".join(lines)


@mcp.tool()
@requires_capability("embedding")
def wiki_reindex(dry_run: bool = False) -> str:
    """Rebuild wiki_pages/wiki_edges tables and re-embed all wiki pages into vector namespaces.

    Walks ``<wiki_root>/pages/**/*.md`` and ``<wiki_root>/_private/**/*.md``,
    deletes existing derived data (wiki_pages, wiki_edges, and wiki/wiki-private
    vector rows), then re-derives everything from the markdown source of truth.
    Idempotent — safe to run repeatedly.

    #35 dim guard: raises an error if the active embedding model's dimension
    differs from the stored vector dimension.

    Args:
        dry_run: When True, scan pages and report counts without writing anything.
    """
    from . import wiki as _wiki
    from pathlib import Path as _Path

    log_tool("wiki_reindex", dry_run=dry_run)
    wiki_root = _Path(_cfg.get("wiki_root") or
                      _Path.home() / ".claude" / "mcp-skill-hub" / "wiki")
    try:
        result = _wiki.reindex(_store, wiki_root, dry_run=dry_run)
    except ValueError as exc:
        return f"wiki_reindex error: {exc}"
    mode = "dry_run" if dry_run else "live"
    return (
        f"wiki_reindex [{mode}]: "
        f"pages={result['pages']} "
        f"edges={result['edges']} "
        f"vectors={result['vectors']}"
    )


@mcp.tool()
@requires_capability("none")
def wiki_status() -> str:
    """Report the current state of the LLM Wiki knowledge layer.

    Returns counts of pages (DB vs disk), edges, orphans (no inbound link),
    dangling edges (unresolved dst), last log.md entry, and drift (pages on
    disk vs vector rows). Stdlib-only — works in no_llm_mode.
    """
    from . import wiki as _wiki
    from pathlib import Path as _Path

    log_tool("wiki_status")
    wiki_root = _Path(_cfg.get("wiki_root") or
                      _Path.home() / ".claude" / "mcp-skill-hub" / "wiki")
    st = _wiki.status(_store, wiki_root)
    lines = [
        f"wiki_status:",
        f"  pages (db={st['pages_db']}, disk={st['pages_disk']}, drift={st['drift']})",
        f"  edges={st['edges']} dangling={st['dangling_edges']} orphans={st['orphans']}",
        f"  vec_rows={st['vec_rows']}",
    ]
    if st["last_log"]:
        lines.append(f"  last_log: {st['last_log']}")
    if st["drift"] > 0:
        lines.append("  WARNING: drift detected — run wiki_reindex to reconcile")
    return "\n".join(lines)


@mcp.tool()
@requires_capability("none")
def wiki_lint() -> str:
    """Deterministic structural lint of the wiki vault. No LLM, stdlib-only.

    Checks three categories:
    - dangling_edges: wiki_edges whose destination page does not exist.
    - orphans: pages with no inbound resolved wikilink.
    - stale_pages: source pages whose underlying file hash differs from the
      stored source_hash (i.e. the original file changed since last ingest).

    Returns counts and a capped sample of offending slugs for each category.
    Run wiki_reindex first so the DB reflects the current vault state.
    """
    from . import wiki as _wiki
    from pathlib import Path as _Path

    log_tool("wiki_lint")
    wiki_root = _Path(_cfg.get("wiki_root") or
                      _Path.home() / ".claude" / "mcp-skill-hub" / "wiki")
    result = _wiki.lint(_store, wiki_root)
    lines = [
        f"wiki_lint: {result['total_issues']} issue(s) found",
        f"  dangling_edges={result['dangling_edges']} "
        f"orphans={result['orphans']} "
        f"stale_pages={result['stale_pages']}",
    ]
    if result["dangling_sample"]:
        sample = ", ".join(result["dangling_sample"][:10])
        lines.append(f"  dangling: {sample}")
    if result["orphan_sample"]:
        sample = ", ".join(result["orphan_sample"][:10])
        lines.append(f"  orphans: {sample}")
    if result["stale_sample"]:
        sample = ", ".join(result["stale_sample"][:10])
        lines.append(f"  stale: {sample}")
    if result["total_issues"] == 0:
        lines.append("  OK — no structural issues detected")
    return "\n".join(lines)


@mcp.tool()
@requires_capability("none")
def wiki_migrate(dry_run: bool = True) -> str:
    """Migrate auto-memory markdown files to wiki source pages (mechanical, no LLM).

    Discovers public auto-memory files via iter_user_memory_files() and private
    files under private/ subdirs, then writes one ``source`` page per file under
    ``<wiki_root>/pages/source/`` (public) or ``<wiki_root>/_private/<scope>/``
    (private).  One ``project/<project>.md`` index page is written per project.

    Idempotent — files already present in any page's ``source_refs`` are skipped.
    After a non-dry-run, call wiki_reindex to rebuild the vector index.

    Args:
        dry_run: When True (default), scan and report counts without writing.
    """
    from . import wiki as _wiki
    from pathlib import Path as _Path

    log_tool("wiki_migrate", dry_run=dry_run)
    wiki_root = _Path(_cfg.get("wiki_root") or
                      _Path.home() / ".claude" / "mcp-skill-hub" / "wiki")
    try:
        result = _wiki.migrate(_store, wiki_root, dry_run=dry_run)
    except Exception as exc:
        return f"wiki_migrate error: {exc}"

    if dry_run:
        lines = [
            f"wiki_migrate [dry_run]: would_write={result['would_write']}",
            f"  public={result['public']} private={result['private']}",
            f"  skipped_already_converted={result['skipped_already_converted']}",
        ]
        if result["collisions"]:
            lines.append(f"  collisions: {', '.join(result['collisions'][:5])}")
        lines.append("  (dry_run: no files written)")
        return "\n".join(lines)

    lines = [
        f"wiki_migrate [live]: written={result['written']}",
        f"  public={result['public']} private={result['private']}",
        f"  skipped={result['skipped']} index_pages={result['index_pages']}",
        "  Run wiki_reindex to rebuild the vector index.",
    ]
    return "\n".join(lines)


@mcp.tool()
@requires_capability("none")
def wiki_scan() -> str:
    """Auto-select wiki source pages needing distillation and enqueue them.

    Deterministic — no LLM, no token spend. Finds undistilled source pages (the
    migration backfill) and stale ones (underlying file changed), then upserts
    them into the approval queue as ``pending``. This is the *automatic source
    selection* step: it picks what to ingest so you never hand-pick files.

    Nothing is distilled here. Approve rows with ``wiki_queue_decision``, then
    run ``wiki_ingest`` to spend tokens. Stdlib + DB only — works in no_llm_mode.
    """
    from . import wiki as _wiki
    from pathlib import Path as _Path

    log_tool("wiki_scan")
    wiki_root = _Path(_cfg.get("wiki_root") or
                      _Path.home() / ".claude" / "mcp-skill-hub" / "wiki")
    summary = _wiki.scan_and_enqueue(_store, wiki_root)
    lines = [
        f"wiki_scan: scanned={summary['scanned']}",
        f"  queue: pending={summary['pending']} approved={summary['approved']} "
        f"done={summary['done']} skipped={summary['skipped']}",
        f"  est_calls(pending+approved)={summary['total_est_calls']}",
    ]
    for c in summary["queue"][:10]:
        title = c["title"] if c["scope"] != "private" else "(private)"
        lines.append(f"  - {c['slug']} [{c['reason']}] {title}")
    extra = len(summary["queue"]) - 10
    if extra > 0:
        lines.append(f"  ... and {extra} more")
    return "\n".join(lines)


@mcp.tool()
@requires_capability("embedding")
def wiki_query(query: str, top_k: int = 5) -> str:
    """Query the LLM Wiki: hybrid vector + index.md lexical ranking.

    Returns the top matching pages with their bodies and ``source_refs``
    provenance. Private pages are included only for authorized scopes (the
    local operator's configured private scopes).
    """
    from . import wiki as _wiki
    from pathlib import Path as _Path

    log_tool("wiki_query")
    wiki_root = _Path(_cfg.get("wiki_root") or
                      _Path.home() / ".claude" / "mcp-skill-hub" / "wiki")
    authorized = _wiki_authorized_scopes()
    out = _wiki.query(_store, wiki_root, query, top_k=top_k,
                      authorized_scopes=authorized)
    results = out["results"]
    if not results:
        return f"wiki_query: no matches for {query!r}"
    lines = [f"wiki_query: {len(results)} result(s) for {query!r}"]
    for r in results:
        refs = ", ".join(r["source_refs"][:3]) if r["source_refs"] else "-"
        lines.append(
            f"\n## [[{r['slug']}]] — {r['title']} "
            f"(type={r['type']}, scope={r['scope']}, score={r['score']})\n"
            f"{r['body'][:1500]}\n(source_refs: {refs})"
        )
    return "\n".join(lines)


@mcp.tool()
@requires_capability("llm")
def wiki_file_answer(query: str, top_k: int = 5) -> str:
    """Query the wiki and synthesize a cited answer from the full page bodies.

    Runs a hybrid vector + index.md search (same as wiki_query), reads the
    full markdown bodies of the top-k hits from disk, then asks the local LLM
    to produce a concise answer that cites each claim with [[slug]] references.

    Fails gracefully when no LLM is available: returns the raw top hits with
    a note instead of raising.

    Args:
        query: Natural-language question to answer from the wiki.
        top_k: Number of wiki pages to retrieve before synthesis (default 5).
    """
    from . import wiki as _wiki
    from pathlib import Path as _Path

    log_tool("wiki_file_answer", query=query[:80], top_k=top_k)
    wiki_root = _Path(_cfg.get("wiki_root") or
                      _Path.home() / ".claude" / "mcp-skill-hub" / "wiki")
    authorized = _wiki_authorized_scopes()
    out = _wiki.file_answer(
        _store, wiki_root, query,
        top_k=top_k, authorized_scopes=authorized,
    )
    lines = [
        f"wiki_file_answer: {len(out['results'])} source(s) for {query!r}",
        "",
        out["answer"],
    ]
    if out["sources"]:
        lines += ["", "Sources: " + ", ".join(f"[[{s}]]" for s in out["sources"])]
    return "\n".join(lines)


def _wiki_authorized_scopes() -> list[str]:
    """Scopes the local operator may write/read — the union of configured
    private scopes (this is the user's own machine and vault)."""
    scopes: set[str] = set()
    cfg = _cfg.get("wiki_private_scopes") or {}
    if isinstance(cfg, dict):
        for v in cfg.values():
            if isinstance(v, list):
                scopes.update(v)
    return sorted(scopes)


@mcp.tool()
@requires_capability("none")
def wiki_queue_decision(slug: str, decision: str) -> str:
    """Approve or skip a wiki ingest candidate.

    ``decision`` is 'approve' or 'skip'. Approval is the gate — only approved
    rows may be ingested live by ``wiki_ingest``. Skipped rows are excluded from
    future ``wiki_scan`` runs until their source changes.
    """
    from . import wiki as _wiki

    log_tool("wiki_queue_decision", decision=decision)
    out = _wiki.queue_decision(_store, slug, decision)
    tail = out.get("decision") or out.get("reason") or ""
    return f"wiki_queue_decision: {out.get('status')} {slug} -> {tail}"


@mcp.tool()
@requires_capability("llm")
def wiki_ingest(slug: str = "", dry_run: bool = True,
                approve_all: bool = False, limit: int = 0) -> str:
    """Distill approved wiki source pages into entity/concept pages via the LLM.

    Automatic selection happens in ``wiki_scan``; this tool spends tokens only
    on rows you have approved (``wiki_queue_decision``).

    Args:
        slug: Ingest one queued source page. Live writes require it be approved.
        dry_run: When True (default), return the proposed diff and write nothing.
        approve_all: Batch-ingest every approved row up to ``limit``.
        limit: Batch cost cap; 0 → config ``wiki_ingest_batch_limit`` (default 10).
    """
    from . import wiki as _wiki
    from pathlib import Path as _Path

    log_tool("wiki_ingest", dry_run=dry_run, approve_all=approve_all)
    wiki_root = _Path(_cfg.get("wiki_root") or
                      _Path.home() / ".claude" / "mcp-skill-hub" / "wiki")
    authorized = _wiki_authorized_scopes()

    if approve_all:
        lim = limit or int(_cfg.get("wiki_ingest_batch_limit") or 10)
        out = _wiki.ingest_approved(_store, wiki_root, limit=lim,
                                    authorized_scopes=authorized, dry_run=dry_run)
        mode = "dry_run" if dry_run else "live"
        return (f"wiki_ingest [batch {mode}]: processed={out['processed']} "
                f"ok={out['ok']} limit={out['limit']}")

    if not slug:
        return "wiki_ingest error: provide slug=... or approve_all=True"

    out = _wiki.ingest_queued(_store, wiki_root, slug,
                              authorized_scopes=authorized, dry_run=dry_run)
    status = out.get("status")
    if status == "ok":
        return (f"wiki_ingest [live]: {slug} status=ok "
                f"written={out.get('written')} vectors={out.get('vectors')}")
    if status == "dry_run":
        return (f"wiki_ingest [dry_run]: {slug} "
                f"pages={len(out.get('pages', []))} (nothing written)")
    return f"wiki_ingest: {slug} status={status} {out.get('reason', '')}"


def main() -> None:
    import sys
    log_banner()
    # Print log path to stderr (Claude console) — stdout is MCP transport
    from .activity_log import LOG_FILE
    print(f"[Skill Hub] Logs: tail -100f {LOG_FILE}", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
