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


def _get_cpu_info() -> int:
    """Get CPU core count for display."""
    import os
    return os.cpu_count() or 0

SETTINGS_PATH = Path.home() / ".claude" / "settings.json"

_store = SkillStore()

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
}

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
    """
    Search skills by semantic similarity to the query.
    Returns full content of the top matching skills — load them directly.

    Args:
        query:      Natural language description of the current task or topic.
        top_k:      Number of skills to load with full content (default 5).
                    Additional matches above the threshold are listed but not loaded.
        use_rerank: If True, use deepseek-r1:1.5b to re-rank results for higher
                    precision (slower, ~2-5s extra per candidate).
    """
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

    parts: list[str] = [header]
    for c in loaded:
        parts.append(
            f"<!-- skill: {c['id']} -->\n{c['content'].strip()}"
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
    """
    Suggest plugins (including disabled ones) that match the current task.
    Combines embedding similarity, teaching rules, and session history.

    Args:
        query: Task description. Leave empty to reuse the last search query.
    """
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
    """
    Record whether a skill was helpful, improving future search rankings.

    Args:
        skill_id: The skill id (e.g. "superpowers:brainstorm") or plugin id.
        helpful:  True if it guided useful work; False if it was a mismatch.
        query:    The original search query. Leave empty to reuse the last search.
    """
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
    """
    Add a persistent rule that maps task patterns to plugins or skills.
    The rule is embedded and matched semantically against future queries.

    Examples:
        teach(rule="when I give a URL to check", suggest="chrome-devtools-mcp")
        teach(rule="working on Terraform infrastructure", suggest="terraform")
        teach(rule="debugging CSS or layout", suggest="chrome-devtools-mcp")
        teach(rule="writing a Telegram bot", suggest="telegram")

    Args:
        rule:    Natural language description of when this suggestion applies.
        suggest: Plugin short name or skill id to suggest.
    """
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
    """
    Remove a teaching rule by its ID.

    Args:
        teaching_id: The teaching ID (shown by list_teachings).
    """
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
    """
    Record that a tool was used in this session (for passive learning).
    Called by the Stop hook to log which plugins were actually used.

    Args:
        tool_name: The MCP tool that was called (e.g. "take_screenshot").
        plugin_id: The plugin that owns the tool (e.g. "chrome-devtools-mcp@...").
    """
    log_tool("log_session", tool_name=tool_name, plugin_id=plugin_id)
    _store.log_session_tool(
        session_id=_session["id"],
        query=_session.get("topic", ""),
        query_vector=_session.get("topic_vector") or None,
        tool_used=tool_name,
        plugin_id=plugin_id or None,
    )
    return f"Logged: {tool_name} → {plugin_id or '(unknown plugin)'}"


# ---------------------------------------------------------------------------
# Tasks (cross-session context)


@mcp.tool()
def save_task(
    title: str,
    summary: str,
    context: str = "",
    tags: str = "",
) -> str:
    """
    Save an open task/discussion for retrieval in future sessions.
    The task stays "open" until you close_task() it.

    Args:
        title:   Short title (e.g. "MCP skill hub development").
        summary: What was discussed, decided, or is in progress.
        context: Extra context — plans, key decisions, file paths, etc.
        tags:    Comma-separated tags for filtering (e.g. "mcp,ollama,sqlite").
    """
    log_tool("save_task", title=title, tags=tags)
    if not ollama_available(EMBED_MODEL):
        return f"Ollama model '{EMBED_MODEL}' not found. Run: ollama pull {EMBED_MODEL}"

    vector = embed(f"{title}: {summary}")
    tid = _store.save_task(
        title=title, summary=summary, vector=vector,
        context=context, tags=tags, session_id=_session["id"],
    )
    return f"Task #{tid} saved (open): \"{title}\"\nWill surface in future search_context() calls."


@mcp.tool()
def close_task(task_id: int, summary: str = "") -> str:
    """
    Close a task with LLM-compacted summary. Uses the local deepseek-r1:1.5b
    to distill the conversation into a compact digest (~200 tokens).

    Args:
        task_id: Task ID from list_tasks().
        summary: Optional final summary. If empty, compacts the existing summary.
    """
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
    return (
        f"Task #{task_id} closed and compacted.\n"
        f"Title: {digest.get('title', 'N/A')}\n"
        f"Summary: {digest.get('summary', 'N/A')}\n"
        f"Tags: {digest.get('tags', 'N/A')}\n"
        f"Decisions: {digest.get('decisions', [])}"
    )


@mcp.tool()
def update_task(task_id: int, summary: str = "", context: str = "",
                tags: str = "") -> str:
    """
    Update an open task with new information.

    Args:
        task_id: Task ID from list_tasks().
        summary: New/appended summary text.
        context: New/appended context.
        tags:    Updated tags.
    """
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
    """Reopen a previously closed task.

    Args:
        task_id: Task ID from list_tasks(status="closed").
    """
    log_tool("reopen_task", task_id=task_id)
    if _store.reopen_task(task_id):
        return f"Task #{task_id} reopened."
    return f"Task #{task_id} not found."


@mcp.tool()
def list_tasks(status: str = "open") -> str:
    """
    List tasks by status.

    Args:
        status: "open", "closed", or "all".
    """
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
def search_context(query: str, top_k: int = 5) -> str:
    """
    Unified search across skills, open tasks, and teachings.
    Returns a combined view of relevant context for the current task.
    Use this at the start of a session to load everything relevant.

    Args:
        query: Natural language description of the task.
        top_k: Max results per category.
    """
    log_tool("search_context", query=query, top_k=top_k)
    if not ollama_available(EMBED_MODEL):
        return f"Ollama model '{EMBED_MODEL}' not found."

    query_vector = embed(query)

    # Update session topic
    if not _session["topic"]:
        _session["topic"] = query
        _session["topic_vector"] = query_vector

    _last_search_state["query"] = query
    _last_search_state["vector"] = query_vector

    parts: list[str] = []

    # 1. Open tasks
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

    # 2. Closed tasks (compact digests)
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
    skills = _store.search(query_vector, top_k=top_k)
    if skills:
        skill_lines = [f"- {s['id']}: {s['description'][:100]}" for s in skills]
        parts.append("## Matching Skills\n\n" + "\n".join(skill_lines))
        _last_search_state["skills"] = [s["id"] for s in skills]

    # 4. Plugin suggestions
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

    if not parts:
        return "No relevant context found. Try index_skills() and index_plugins() first."

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Management


@mcp.tool()
def index_skills() -> str:
    """
    Rebuild the skill index from all Claude Code plugin directories.
    Run this once after installing or updating plugins.
    Requires Ollama with nomic-embed-text: ollama pull nomic-embed-text
    """
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
    """
    Index plugin descriptions for suggest_plugins().
    Reads enabledPlugins from settings.json and indexes their descriptions
    from the plugin manifest files.
    """
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
    """
    List all indexed skills with their descriptions.

    Args:
        plugin: Optional plugin name to filter by (e.g. "superpowers").
    """
    rows = _store.list_skills()
    if plugin:
        rows = [r for r in rows if r["plugin"] == plugin]
    if not rows:
        return "No skills indexed. Run index_skills() first."
    lines = [f"- {r['id']}: {r['description'] or '(no description)'}" for r in rows]
    return f"{len(lines)} skills:\n" + "\n".join(lines)


@mcp.tool()
def toggle_plugin(plugin_name: str, enabled: bool) -> str:
    """
    Enable or disable a plugin in ~/.claude/settings.json.
    Change takes effect on the NEXT Claude Code session restart.

    Args:
        plugin_name: Plugin key as it appears in enabledPlugins
                     (e.g. "superpowers@claude-plugins-official").
        enabled:     True to enable, False to disable.
    """
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
    """
    View or update Skill Hub configuration.
    Config file: ~/.claude/mcp-skill-hub/config.json

    Without arguments: show current config.
    With key+value: update a setting.

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
    """
    Use the local LLM to analyze all memory files and recommend pruning.

    Reads every .md file in the Claude auto-memory directory, sends them
    to the local reasoning model, and returns a report with recommendations:
    KEEP, PRUNE (stale/done), COMPACT (too verbose), or MERGE (duplicates).

    Args:
        dry_run: If True (default), only report recommendations.
                 If False, apply PRUNE actions (delete files + update MEMORY.md).
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
def status() -> str:
    """
    Show the health status of Skill Hub and its dependencies.

    Checks:
    - MCP server running (always true if this tool responds)
    - Ollama reachable
    - Embedding model available
    - Reasoning model available
    - Hook configured in settings.json
    - Token profiling on/off
    - Skill and task counts in the database
    """
    import httpx

    cfg = _cfg.load_config()
    ollama_base = cfg.get("ollama_base", "http://localhost:11434")
    embed_model = cfg.get("embed_model", "nomic-embed-text")
    reason_model = cfg.get("reason_model", "deepseek-r1:1.5b")

    lines: list[str] = ["=== Skill Hub Status ===\n"]

    # MCP server
    lines.append("MCP server:      ✓ running (this response proves it)")

    # Ollama + models
    try:
        resp = httpx.get(f"{ollama_base}/api/tags", timeout=5.0)
        available = [m["name"] for m in resp.json().get("models", [])]
        lines.append(f"Ollama:          ✓ reachable at {ollama_base}")

        embed_ok = any(embed_model in m for m in available)
        lines.append(
            f"Embed model:     {'✓' if embed_ok else '✗'} {embed_model}"
            + ("" if embed_ok else f"  ← run: ollama pull {embed_model}")
        )
        reason_ok = any(reason_model in m for m in available)
        lines.append(
            f"Reason model:    {'✓' if reason_ok else '✗'} {reason_model}"
            + ("" if reason_ok else f"  ← run: ollama pull {reason_model}")
        )
        if available:
            lines.append(f"Other models:    {', '.join(m.split(':')[0] for m in available)}")
    except Exception as exc:
        lines.append(f"Ollama:          ✗ NOT reachable at {ollama_base} ({exc})")
        lines.append(f"  ← start Ollama: brew services start ollama")

    # Hook
    hook_configured = False
    if SETTINGS_PATH.exists():
        try:
            settings = json.loads(SETTINGS_PATH.read_text())
            hooks = settings.get("hooks", {}).get("UserPromptSubmit", [])
            for group in hooks:
                for h in group.get("hooks", []):
                    if "intercept-task-commands" in h.get("command", ""):
                        hook_configured = True
        except Exception:
            pass
    hook_enabled = cfg.get("hook_enabled", True)
    hook_status = (
        "✓ configured and enabled" if hook_configured and hook_enabled
        else ("configured but disabled (hook_enabled=false)" if hook_configured
              else "✗ NOT configured — see README.md#install-the-hook")
    )
    lines.append(f"Hook:            {hook_status}")

    # Token profiling
    profiling = cfg.get("token_profiling", True)
    lines.append(f"Token profiling: {'✓ on' if profiling else '○ off'}"
                 + ("  ← configure(key='token_profiling', value='true') to enable"
                    if not profiling else ""))

    # DB stats
    try:
        rows = _store.list_skills()
        skill_count = len(rows)
        task_rows = _store.list_tasks("all")
        task_count = len(task_rows)
        open_tasks = sum(1 for r in task_rows if r["status"] == "open")
        totals = _store.get_interception_totals()
        saved = totals["total_tokens_saved"] or 0 if totals else 0
        intercepted = totals["total_interceptions"] or 0 if totals else 0
        lines.append(f"\nDatabase ({_store._conn.execute('PRAGMA database_list').fetchone()[2]}):")
        lines.append(f"  Skills indexed:    {skill_count}")
        lines.append(f"  Tasks:             {task_count} ({open_tasks} open)")
        lines.append(f"  Intercepted cmds:  {intercepted} (~{saved:,} tokens saved)")
    except Exception as exc:
        lines.append(f"\nDatabase: error — {exc}")

    # Config summary
    lines.append(f"\nConfig ({_cfg.CONFIG_PATH}):")
    lines.append(f"  embed_model={embed_model}, reason_model={reason_model}")
    lines.append(f"  search_top_k={cfg.get('search_top_k')}, hook_timeout={cfg.get('hook_timeout_seconds')}s")

    # Context usage — estimate token cost of always-loaded files
    lines.append("\n=== Context Usage (estimated) ===\n")

    def _est_tokens(path: Path) -> int:
        """Rough estimate: 1 token ≈ 4 chars."""
        try:
            return max(1, len(path.read_text(encoding="utf-8", errors="replace")) // 4)
        except OSError:
            return 0

    context_files = [
        ("User    ", Path.home() / ".claude" / "CLAUDE.md"),
        ("Project ", Path.home() / "work" / "code" / "CLAUDE.md"),
        ("AutoMem ", Path.home() / ".claude" / "projects" /
         "-Users-ccancellieri-work-code" / "memory" / "MEMORY.md"),
    ]
    # Also check the working-directory CLAUDE.md dynamically
    import os
    cwd_claude = Path(os.getcwd()) / "CLAUDE.md"
    if cwd_claude.exists() and cwd_claude not in [f for _, f in context_files]:
        context_files.append(("Cwd     ", cwd_claude))

    total_ctx = 0
    for label, fpath in context_files:
        if fpath.exists():
            tok = _est_tokens(fpath)
            total_ctx += tok
            size_kb = fpath.stat().st_size / 1024
            lines.append(f"  {label}  ~{tok:>5} tokens  ({size_kb:.1f} KB)  {fpath}")
        else:
            lines.append(f"  {label}  (not found)  {fpath}")

    lines.append(f"\n  Total always-loaded:  ~{total_ctx:,} tokens")

    # Breakdown by category
    lines.append("\n  Breakdown by category:")
    mem_path = Path.home() / ".claude" / "projects" / \
               "-Users-ccancellieri-work-code" / "memory"
    if mem_path.exists():
        cats: dict[str, int] = {}
        for mf in sorted(mem_path.glob("*.md")):
            if mf.name == "MEMORY.md":
                continue
            prefix = mf.stem.split("_")[0]
            cats[prefix] = cats.get(prefix, 0) + _est_tokens(mf)
        for cat, tok in sorted(cats.items(), key=lambda x: -x[1]):
            lines.append(f"    {cat:<12} ~{tok:>5} tokens  (detail files, loaded on demand)")

    # System resource pressure
    lines.append("\n=== System Resources ===\n")
    s = snapshot(force=True)
    pressure_icon = {"IDLE": "🟢", "LOW": "🟡", "MODERATE": "🟠", "HIGH": "🔴"}
    lines.append(f"  Pressure:  {pressure_icon.get(s.pressure.name, '?')} {s.pressure.name}")
    lines.append(f"  CPU load:  {s.cpu_load_1m:.0%} (normalized to {_get_cpu_info()} cores)")
    lines.append(f"  Memory:    {s.memory_used_pct:.0%} used, {s.memory_available_mb}MB available / {s.total_memory_mb}MB total")
    lines.append(f"  LLM gates: triage={'on' if s.pressure.value <= 2 else 'SKIP'}, "
                 f"precompact={'on' if s.pressure.value <= 1 else 'SKIP'}, "
                 f"digest={'on' if s.pressure.value <= 1 else 'SKIP'}, "
                 f"optimize={'on' if s.pressure.value == 0 else 'SKIP'}")

    # Memory optimization tips
    lines.append("\n=== Memory Optimization Tips ===\n")
    mem_index_tok = _est_tokens(
        Path.home() / ".claude" / "projects" /
        "-Users-ccancellieri-work-code" / "memory" / "MEMORY.md"
    )
    tips = []
    if mem_index_tok > 1500:
        tips.append(
            f"  ✦ MEMORY.md index is ~{mem_index_tok} tokens — prune entries for "
            f"completed/stale projects. Each removed line saves ~10-20 tokens every session."
        )
    user_tok = _est_tokens(Path.home() / ".claude" / "CLAUDE.md")
    if user_tok > 600:
        tips.append(
            f"  ✦ User CLAUDE.md (~{user_tok} tokens) — move static reference content "
            f"(model lists, port tables) to MCP: save_task() or teach() rules, then delete from CLAUDE.md."
        )
    proj_tok = _est_tokens(Path.home() / "work" / "code" / "CLAUDE.md")
    if proj_tok > 400:
        tips.append(
            f"  ✦ Project CLAUDE.md (~{proj_tok} tokens) — the routing table can be "
            f"trimmed once projects are stable. Archive inactive project pointers."
        )
    tips.append(
        "  ✦ Use search_context() at session start instead of pre-loading project memory files."
    )
    tips.append(
        "  ✦ Closed tasks are compacted to ~200 tokens and searchable — "
        "prefer close_task() over growing MEMORY.md with raw notes."
    )
    if not tips:
        tips.append("  Context looks lean — no obvious savings available.")
    lines.extend(tips)

    return "\n".join(lines)


@mcp.tool()
def token_stats() -> str:
    """
    Show token savings from hook interceptions.

    Displays estimated Claude API tokens saved by the UserPromptSubmit hook
    intercepting task commands before they reach Claude. Enable/disable
    profiling with configure(key='token_profiling', value='true|false').
    """
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
def list_models() -> str:
    """
    List Ollama models installed on this machine, showing which ones are
    configured for Skill Hub and recommendations by hardware tier.

    The two model roles:
    - embed_model  — converts text to vectors for semantic search (always running)
    - reason_model — LLM used for re-ranking results, compacting tasks, and
                     classifying hook commands (runs only when needed)

    Use pull_model() to download a new model, then configure() to activate it.
    """
    import httpx

    cfg = _cfg.load_config()
    ollama_base = cfg.get("ollama_base", "http://localhost:11434")
    current_embed = cfg.get("embed_model", "nomic-embed-text")
    current_reason = cfg.get("reason_model", "deepseek-r1:1.5b")

    lines = ["=== Ollama Models ===\n"]

    # What the two roles do
    lines.append("Model roles:")
    lines.append("  embed_model  — Converts text to semantic vectors.")
    lines.append("                 Used for ALL searches (search_skills, suggest_plugins,")
    lines.append("                 search_context). Runs once per query, very fast.")
    lines.append("                 Current: " + current_embed)
    lines.append("")
    lines.append("  reason_model — Small local LLM. Used for:")
    lines.append("                 • Re-ranking search results (use_rerank=True)")
    lines.append("                 • Compacting tasks on close_task()")
    lines.append("                 • Classifying hook commands (save/close/list)")
    lines.append("                 Runs only when needed, heavier than embed.")
    lines.append("                 Current: " + current_reason)
    lines.append("")

    # Installed models
    try:
        resp = httpx.get(f"{ollama_base}/api/tags", timeout=5.0)
        installed = resp.json().get("models", [])
        if installed:
            lines.append("Installed models:")
            for m in installed:
                name = m["name"]
                size_gb = m.get("size", 0) / 1_073_741_824
                role = ""
                if current_embed in name:
                    role = "  ← embed_model (active)"
                elif current_reason in name:
                    role = "  ← reason_model (active)"
                lines.append(f"  {name:<35} {size_gb:.1f} GB{role}")
        else:
            lines.append("No models installed yet.")
    except Exception as exc:
        lines.append(f"Ollama not reachable ({exc})")

    # Recommendations table
    lines.append("\nReasoning models (reason_model):")
    lines.append("  Model                  Size    RAM     Notes")
    lines.append("  deepseek-r1:1.5b       1.1 GB  8 GB    Minimal, fast hook classification")
    lines.append("  deepseek-r1:3b         2.1 GB  8 GB    Good quality/speed for 8-16 GB RAM")
    lines.append("  deepseek-r1:7b         4.7 GB  16 GB   Recommended — MacBook Pro sweet spot")
    lines.append("  deepseek-r1:8b         5.2 GB  16 GB   Llama3 architecture variant")
    lines.append("  deepseek-r1:14b        9 GB    32 GB   Best quality for 32 GB RAM")
    lines.append("  qwen2.5-coder:3b       2 GB    8 GB    Code-focused, fast")
    lines.append("  qwen2.5-coder:7b       4.7 GB  16 GB   Code-focused reasoning")
    lines.append("  phi4-mini              2.5 GB  8 GB    Microsoft, strong for its size")
    lines.append("  gemma3:4b              3.3 GB  8 GB    Google, strong instruction following")
    lines.append("")
    lines.append("Embedding models (embed_model):")
    lines.append("  nomic-embed-text       274 MB  any     Default — fast, good quality")
    lines.append("  mxbai-embed-large      669 MB  16 GB+  Higher quality embeddings")
    lines.append("")
    lines.append("To install a model:   pull_model(model='deepseek-r1:7b')")
    lines.append("To activate a model:  configure(key='reason_model', value='deepseek-r1:7b')")
    lines.append("                      configure(key='embed_model', value='mxbai-embed-large')")
    lines.append("After changing embed_model, run index_skills() to rebuild vectors.")

    return "\n".join(lines)


@mcp.tool()
def pull_model(model: str) -> str:
    """
    Pull (download) an Ollama model to this machine.

    This runs `ollama pull <model>` and streams progress. The model becomes
    available immediately after for configure(key='reason_model'/'embed_model').

    Common models:
        nomic-embed-text    — 274 MB — default embed model
        mxbai-embed-large   — 669 MB — higher-quality embeddings
        deepseek-r1:1.5b    — 1.1 GB — fast reasoning, 8 GB RAM
        deepseek-r1:7b      — 4.7 GB — good quality, 16 GB RAM
        deepseek-r1:14b     — 9 GB   — best quality, 32 GB RAM
        qwen2.5-coder:7b    — 4.7 GB — code-focused reasoning

    Args:
        model: Model name as shown on hub.ollama.ai (e.g. "deepseek-r1:7b")
    """
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
def exhaustion_save(context: str = "") -> str:
    """
    Auto-save the current session when Claude is exhausted or rate-limited.

    Uses the local LLM to generate a structured task save with title, summary,
    decisions, next steps, and files modified. The saved task can be resumed
    later with search_context() or list_tasks().

    Call this when you detect Claude is running low on context or quota.
    Also available as /exhaustion-save slash command (0 Claude tokens).

    Args:
        context: Description of current work state. If empty, uses recent
                 session messages from the hook pipeline.
    """
    from .cli import _cmd_exhaustion_save
    return _cmd_exhaustion_save(context)


def main() -> None:
    import sys
    log_banner()
    # Print log path to stderr (Claude console) — stdout is MCP transport
    from .activity_log import LOG_FILE
    print(f"[Skill Hub] Logs: tail -100f {LOG_FILE}", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
