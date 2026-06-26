"""Dashboard KPI card for the wiki-to-skill-generator plugin.

Returns one section showing generated skills count and pending candidates.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

_PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT.parent.parent / "src"))


def _get_hub_db_path() -> Path:
    from skill_hub import config as _cfg
    return Path(_cfg.get("db_path") or Path.home() / ".claude" / "mcp-skill-hub" / "skill_hub.db")


def _load_plugin_config() -> dict[str, Any]:
    plugin_json = _PLUGIN_ROOT / "plugin.json"
    try:
        return json.loads(plugin_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def get_sections() -> list[dict]:
    db_path = _get_hub_db_path()
    config = _load_plugin_config()
    plugin_config = config.get("config", {})
    min_access = plugin_config.get("min_access_count", 3)
    exclude_types = plugin_config.get("exclude_types", ["source", "overview"])

    generated_count = 0
    pending_count = 0

    if db_path.exists():
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row

                # Count generated skills
                skills_row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM plugin_wiki_skills"
                ).fetchone()
                generated_count = skills_row["cnt"] if skills_row else 0

                # Count pending candidates
                placeholders = ",".join("?" * len(exclude_types))
                pending_row = conn.execute(
                    f"""
                    SELECT COUNT(*) as cnt
                    FROM vectors v
                    LEFT JOIN plugin_wiki_skills s ON v.doc_id = s.wiki_slug
                    LEFT JOIN wiki_pages p ON v.doc_id = p.slug
                    WHERE v.namespace IN ('wiki', 'wiki-private')
                      AND v.access_count >= ?
                      AND s.id IS NULL
                      AND p.type NOT IN ({placeholders})
                    """,
                    (min_access, *exclude_types),
                ).fetchone()
                pending_count = pending_row["cnt"] if pending_row else 0
        except sqlite3.Error:
            pass

    return [
        {
            "id": "wiki-skill-generator",
            "title": "Wiki Skill Generator",
            "order": 85,
            "template": "dashboard_kpi.html",
            "context": {
                "generated_count": generated_count,
                "pending_count": pending_count,
                "min_access": min_access,
            },
        }
    ]
