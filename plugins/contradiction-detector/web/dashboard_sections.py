"""Dashboard KPI card for the contradiction-detector plugin.

Returns one section showing pending contradictions count and last scan info.
Renders via the shared ``kpi_card`` macro.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path.home() / ".claude" / "mcp-skill-hub" / "skill_hub.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_sections() -> list[dict]:
    pending = 0
    resolved = 0
    last_run = None

    if DB_PATH.exists():
        try:
            with _get_conn() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM plugin_contradiction_findings WHERE resolution_status = 'pending'"
                ).fetchone()
                pending = row[0] if row else 0

                row = conn.execute(
                    "SELECT COUNT(*) FROM plugin_contradiction_findings WHERE resolution_status = 'resolved'"
                ).fetchone()
                resolved = row[0] if row else 0

                row = conn.execute(
                    """
                    SELECT started_at, status, contradictions_found
                    FROM plugin_contradiction_runs
                    ORDER BY started_at DESC LIMIT 1
                    """
                ).fetchone()
                if row:
                    last_run = {
                        "started_at": row["started_at"],
                        "status": row["status"],
                        "contradictions_found": row["contradictions_found"],
                    }
        except sqlite3.Error:
            pass

    return [
        {
            "id": "contradiction-detector",
            "title": "Contradiction Detector",
            "order": 70,
            "template": "dashboard_kpi.html",
            "context": {
                "pending": pending,
                "resolved": resolved,
                "last_run": last_run,
            },
        }
    ]
