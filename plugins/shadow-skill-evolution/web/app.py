"""FastAPI sub-app for the /shadow-skill-evolution mount.

Provides a dashboard to review and approve/reject skill proposals.

``get_app()`` returns a FastAPI instance that mcp-skill-hub mounts under the
path declared in ``plugin.json -> web_mount.mount`` (``/shadow-skill-evolution``).
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from jinja2 import FileSystemLoader
from starlette.templating import Jinja2Templates

HERE = Path(__file__).parent
PLUGIN_ROOT = HERE.parent
TEMPLATES_DIR = HERE / "templates"

PLUGIN_DB_PATH = Path.home() / ".claude" / "mcp-skill-hub" / "plugins" / "shadow_skill_evolution.db"
LOCAL_SKILLS_DIR = Path.home() / ".claude" / "local-skills"

if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

_DEFAULT_SHARED = "/Users/ccancellieri/work/code/mcp-skill-hub/src/skill_hub/webapp/templates"


def _build_templates() -> Jinja2Templates:
    import os
    shared = os.environ.get("SKILL_HUB_SHARED_TEMPLATES", _DEFAULT_SHARED)
    search_paths = [str(TEMPLATES_DIR)]
    if shared and Path(shared).exists():
        search_paths.append(shared)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.loader = FileSystemLoader(search_paths)
    return templates


def _get_conn() -> sqlite3.Connection:
    PLUGIN_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(PLUGIN_DB_PATH))
    conn.row_factory = sqlite3.Row
    schema_path = PLUGIN_ROOT / "storage" / "schema.sql"
    if schema_path.exists():
        conn.executescript(schema_path.read_text())
        conn.commit()
    return conn


def get_app() -> FastAPI:
    app = FastAPI(
        title="Shadow Skill Evolution",
        description="Review and approve auto-generated skill proposals",
        docs_url="/docs",
    )
    templates = _build_templates()
    app.state.templates = templates
    app.state.plugin_root = PLUGIN_ROOT

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, status: str = "pending"):
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
                    p["source_chains"] = json.loads(p["source_chains"]) if p["source_chains"] else []
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
            }
        finally:
            conn.close()

        return templates.TemplateResponse(
            "proposals.html",
            {"request": request, "proposals": proposals, "status": status, "stats": stats},
        )

    @app.post("/approve/{proposal_id}")
    def approve_proposal(proposal_id: int):
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM skill_proposals WHERE id = ? AND status = 'pending'",
                (proposal_id,),
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Proposal not found")

            proposal = dict(row)
            skill_name = proposal["name"]

            skill_data = {
                "name": skill_name,
                "title": proposal["title"],
                "description": proposal["description"],
                "triggers": json.loads(proposal["triggers"]) if proposal["triggers"] else [],
                "steps": json.loads(proposal["steps"]) if proposal["steps"] else [],
                "output": f"Executed skill: {proposal['title']}",
            }

            LOCAL_SKILLS_DIR.mkdir(parents=True, exist_ok=True)
            skill_path = LOCAL_SKILLS_DIR / f"{skill_name}.json"
            skill_path.write_text(json.dumps(skill_data, indent=2), encoding="utf-8")

            conn.execute(
                "UPDATE skill_proposals SET status = 'approved', approved_at = datetime('now') "
                "WHERE id = ?",
                (proposal_id,),
            )
            conn.execute(
                "INSERT INTO generated_skills (proposal_id, skill_path, skill_name) "
                "VALUES (?, ?, ?)",
                (proposal_id, str(skill_path), skill_name),
            )
            conn.commit()
        finally:
            conn.close()

        return RedirectResponse(url=f"/shadow-skill-evolution?status=approved", status_code=303)

    @app.post("/reject/{proposal_id}")
    def reject_proposal(proposal_id: int):
        conn = _get_conn()
        try:
            conn.execute(
                "UPDATE skill_proposals SET status = 'rejected', rejected_at = datetime('now') "
                "WHERE id = ? AND status = 'pending'",
                (proposal_id,),
            )
            conn.commit()
        finally:
            conn.close()

        return RedirectResponse(url=f"/shadow-skill-evolution?status=rejected", status_code=303)

    @app.get("/api/proposals")
    def api_list_proposals(status: str = "pending", limit: int = 50):
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT id, name, title, description, cluster_size, similarity_score, created_at "
                "FROM skill_proposals WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
            return JSONResponse([dict(r) for r in rows])
        finally:
            conn.close()

    @app.get("/api/chains")
    def api_list_chains(limit: int = 100):
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT id, chain_hash, tool_sequence, occurrence_count, last_seen_at "
                "FROM tool_chains ORDER BY last_seen_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            chains = []
            for r in rows:
                chain = dict(r)
                try:
                    chain["tool_sequence"] = json.loads(chain["tool_sequence"])
                except json.JSONDecodeError:
                    pass
                chains.append(chain)
            return JSONResponse(chains)
        finally:
            conn.close()

    @app.delete("/api/proposals/{proposal_id}")
    def api_delete_proposal(proposal_id: int):
        conn = _get_conn()
        try:
            conn.execute("DELETE FROM skill_proposals WHERE id = ?", (proposal_id,))
            conn.commit()
            return JSONResponse({"status": "deleted"})
        finally:
            conn.close()

    return app
