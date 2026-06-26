"""``GET /`` — overview page with proposal stats and management."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fastapi import APIRouter, Request

router = APIRouter()

PLUGIN_DB_PATH = (
    Path.home() / ".claude" / "mcp-skill-hub" / "plugins" / "shadow_skill_evolution.db"
)


def _get_conn() -> sqlite3.Connection:
    PLUGIN_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(PLUGIN_DB_PATH))
    conn.row_factory = sqlite3.Row
    schema_path = (
        Path(__file__).resolve().parent.parent.parent / "storage" / "schema.sql"
    )
    if schema_path.exists():
        conn.executescript(schema_path.read_text())
        conn.commit()
    return conn


@router.get("/")
def index(request: Request, status: str = "pending"):
    templates = request.app.state.templates
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM skill_proposals WHERE status = ? "
            "ORDER BY created_at DESC LIMIT 50",
            (status,),
        ).fetchall()
        proposals = [dict(r) for r in rows]
        for p in proposals:
            try:
                p["triggers"] = json.loads(p["triggers"]) if p["triggers"] else []
                p["steps"] = json.loads(p["steps"]) if p["steps"] else []
                p["source_chains"] = (
                    json.loads(p["source_chains"]) if p["source_chains"] else []
                )
            except json.JSONDecodeError:
                pass

        stats = {
            "pending": conn.execute(
                "SELECT COUNT(*) FROM skill_proposals WHERE status='pending'"
            ).fetchone()[0],
            "approved": conn.execute(
                "SELECT COUNT(*) FROM skill_proposals WHERE status='approved'"
            ).fetchone()[0],
            "rejected": conn.execute(
                "SELECT COUNT(*) FROM skill_proposals WHERE status='rejected'"
            ).fetchone()[0],
            "chains": conn.execute(
                "SELECT COUNT(*) FROM tool_chains"
            ).fetchone()[0],
        }
    finally:
        conn.close()

    return templates.TemplateResponse(
        request,
        "page.html",
        {
            "active_tab": "skill-evolution",
            "proposals": proposals,
            "status": status,
            "stats": stats,
        },
    )
