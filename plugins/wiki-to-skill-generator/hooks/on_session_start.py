#!/usr/bin/env python3
"""A3 — on_session_start hook for wiki-to-skill-generator plugin.

Reads plugin_wiki_skills table and suggests relevant wiki-derived skills
to the active session. Can optionally filter by context relevance.

Input (stdin JSON):
{
  "event": "on_session_start",
  "session_id": "...",
  "cwd": "...",  // optional
  "task_id": 123  // optional, if session bound to a task
}

Output (stdout JSON):
{
  "suggestions": [
    {
      "skill_name": "...",
      "skill_path": "...",
      "wiki_slug": "...",
      "relevance": "high|medium|low"
    }
  ],
  "inject": "...optional markdown to inject into session context..."
}
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

_PLUGINDIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PLUGINDIR.parent.parent / "src"))

_log = logging.getLogger(__name__)


def _get_registered_skills(store: Any, limit: int = 10) -> list[dict]:
    """Fetch wiki-derived skills from plugin_wiki_skills table."""
    try:
        db = store.plugin_db("wiki_skills")
        rows = db.fetch_all(
            """
            SELECT wiki_slug, wiki_title, wiki_type, skill_path, skill_id, 
                   access_count, use_count
            FROM plugin_wiki_skills
            ORDER BY access_count DESC, use_count DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(r) for r in rows] if rows else []
    except Exception as exc:
        _log.warning("Failed to fetch registered skills: %s", exc)
        return []


def _update_use_count(store: Any, wiki_slug: str) -> None:
    """Increment use_count for a skill."""
    try:
        db = store.plugin_db("wiki_skills")
        db.execute(
            """
            UPDATE plugin_wiki_skills
            SET use_count = use_count + 1, last_used = datetime('now')
            WHERE wiki_slug = ?
            """,
            (wiki_slug,),
        )
    except Exception as exc:
        _log.warning("Failed to update use count: %s", exc)


def _format_suggestion(skill: dict) -> str:
    """Format a skill suggestion as markdown."""
    return f"- [[{skill['skill_id'] or skill['wiki_slug']}]] — {skill['wiki_title']} (from wiki, {skill['access_count']} accesses)"


def main() -> None:
    """Read event payload, suggest relevant skills."""
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return
    
    event = payload.get("event", "")
    if event != "on_session_start":
        return
    
    try:
        from skill_hub.store import SkillStore
        store = SkillStore()
    except Exception as exc:
        _log.warning("Failed to get store: %s", exc)
        return
    
    skills = _get_registered_skills(store)
    
    if not skills:
        return
    
    # Format suggestions
    lines = ["## Wiki-Derived Skills Available", ""]
    lines.append("The following skills were generated from frequently-accessed wiki pages:")
    lines.append("")
    lines.extend(_format_suggestion(s) for s in skills)
    lines.append("")
    lines.append("Use `skill` tool with the skill name to load detailed guidance.")
    
    # Update use counts
    for skill in skills[:5]:
        _update_use_count(store, skill["wiki_slug"])
    
    try:
        store.close()
    except Exception:
        pass
    
    output = {
        "suggestions": [
            {
                "skill_name": s["skill_id"] or s["wiki_slug"],
                "skill_path": s["skill_path"],
                "wiki_slug": s["wiki_slug"],
                "relevance": "medium",
            }
            for s in skills
        ],
        "inject": "\n".join(lines),
    }
    
    print(json.dumps(output))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
