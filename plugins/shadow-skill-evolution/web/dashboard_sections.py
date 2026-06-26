"""Dashboard KPI card for the skill-evolution plugin.

Returns one section showing pending proposals, approved skills count,
and detected tool-chain patterns. Renders via the shared ``kpi_card`` macro.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

_PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

PLUGIN_DB_PATH = (
    Path.home() / ".claude" / "mcp-skill-hub" / "plugins" / "shadow_skill_evolution.db"
)


def get_sections() -> list[dict]:
    pending = 0
    approved = 0
    chains = 0

    if PLUGIN_DB_PATH.exists():
        try:
            with sqlite3.connect(str(PLUGIN_DB_PATH)) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT COUNT(*) FROM skill_proposals WHERE status='pending'"
                ).fetchone()
                pending = row[0] if row else 0

                row = conn.execute(
                    "SELECT COUNT(*) FROM skill_proposals WHERE status='approved'"
                ).fetchone()
                approved = row[0] if row else 0

                row = conn.execute("SELECT COUNT(*) FROM tool_chains").fetchone()
                chains = row[0] if row else 0
        except sqlite3.Error:
            pass

    return [
        {
            "id": "skill-evolution",
            "title": "Skill Evolution",
            "order": 90,
            "template": "dashboard_kpi.html",
            "context": {
                "pending_proposals": pending,
                "approved_skills": approved,
                "tool_chains": chains,
                "link": "/skill-evolution",
            },
        }
    ]
