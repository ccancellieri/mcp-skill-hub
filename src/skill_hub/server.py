"""
Skill Hub MCP Server
====================
Local stdio MCP server that provides semantic skill search for Claude Code.

Tools
-----
search_skills(query, top_k, use_rerank)
    Embed the query, find similar skills, optionally re-rank with deepseek-r1.
    Returns full skill content of top matches — ready to use inline.

record_feedback(skill_id, query, helpful)
    Record whether a skill was useful for a given query.
    Improves future ranking via feedback boost.

index_skills()
    Rebuild the SQLite index from all plugin directories.
    Run once after installing/updating plugins.

list_skills(plugin)
    List all indexed skills (optionally filtered by plugin).

toggle_plugin(plugin_name, enabled)
    Enable or disable a plugin in ~/.claude/settings.json.
    Takes effect on the next Claude Code session restart.
"""

import json
from pathlib import Path

from fastmcp import FastMCP

from .embeddings import embed, rerank, EMBED_MODEL, RERANK_MODEL, ollama_available
from .indexer import index_all
from .store import SkillStore

SETTINGS_PATH = Path.home() / ".claude" / "settings.json"

_store = SkillStore()

mcp = FastMCP(
    "skill-hub",
    instructions="""
Skill Hub gives you semantic skill search. Workflow:

1. At the start of a conversation (or when a new topic arises), call
   search_skills(query="<describe current task>") to load relevant skills.
2. Read the returned skill content and follow its guidance.
3. After finishing the task, call record_feedback(skill_id, query, helpful=True/False)
   so future searches improve for similar tasks.

Skills not returned by search are NOT in context — keep the context clean.
""",
)


# ---------------------------------------------------------------------------
# Tools


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
    candidates = _store.search(query_vector, top_k=top_k * 2 if use_rerank else top_k)

    if not candidates:
        return "No matching skills found. Run index_skills() if the index is empty."

    if use_rerank and len(candidates) > 1:
        candidates = rerank(query, candidates, model=RERANK_MODEL)[:top_k]
    else:
        candidates = candidates[:top_k]

    # Record that we surfaced these skills (for implicit feedback later)
    _last_search_state["query"] = query
    _last_search_state["vector"] = query_vector
    _last_search_state["skills"] = [c["id"] for c in candidates]

    parts: list[str] = []
    for c in candidates:
        parts.append(
            f"<!-- skill: {c['id']} -->\n{c['content'].strip()}"
        )
    return "\n\n---\n\n".join(parts)


# Transient in-process state so record_feedback can omit query when called
# right after search_skills.
_last_search_state: dict = {"query": "", "vector": [], "skills": []}


@mcp.tool()
def record_feedback(
    skill_id: str,
    helpful: bool,
    query: str = "",
) -> str:
    """
    Record whether a skill was helpful, improving future search rankings.

    Args:
        skill_id: The skill id as shown in search results (e.g. "superpowers:brainstorm").
        helpful:  True if the skill guided useful work; False if it was a mismatch.
        query:    The original search query. Leave empty to reuse the last search query.
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


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
