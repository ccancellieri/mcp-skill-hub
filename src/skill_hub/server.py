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
"""

import json
import uuid
from pathlib import Path

from fastmcp import FastMCP

from .embeddings import (
    embed, rerank, compact, rewrite_query,
    EMBED_MODEL, RERANK_MODEL, ollama_available,
)
from .indexer import index_all
from .store import SkillStore

SETTINGS_PATH = Path.home() / ".claude" / "settings.json"

_store = SkillStore()

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
    top_k: int = 3,
    use_rerank: bool = False,
) -> str:
    """
    Search skills by semantic similarity to the query.
    Returns full content of the top matching skills — load them directly.

    Args:
        query:      Natural language description of the current task or topic.
        top_k:      Number of skills to return (default 3).
        use_rerank: If True, use deepseek-r1:1.5b to re-rank results for higher
                    precision (slower, ~2-5s extra per candidate).
    """
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

    candidates = _store.search(query_vector, top_k=top_k * 2 if use_rerank else top_k)

    if not candidates:
        return "No matching skills found. Run index_skills() if the index is empty."

    if use_rerank and len(candidates) > 1:
        candidates = rerank(query, candidates, model=RERANK_MODEL)[:top_k]
    else:
        candidates = candidates[:top_k]

    _last_search_state["query"] = query
    _last_search_state["vector"] = query_vector
    _last_search_state["skills"] = [c["id"] for c in candidates]

    parts: list[str] = []
    for c in candidates:
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
    for plugin_key in plugins:
        short_name = plugin_key.split("@")[0]
        desc = descriptions.get(short_name, "")
        if not desc:
            desc = f"Plugin: {short_name}"

        _store.upsert_plugin(plugin_key, short_name, desc)
        try:
            vector = embed(f"{short_name}: {desc}")
            _store.upsert_plugin_embedding(plugin_key, EMBED_MODEL, vector)
            indexed += 1
        except Exception as exc:
            errors.append(f"{short_name}: {exc}")

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


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
