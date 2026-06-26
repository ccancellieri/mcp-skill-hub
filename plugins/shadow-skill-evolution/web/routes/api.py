"""REST endpoints: proposals CRUD, approve/reject, analyze trigger."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse

router = APIRouter(prefix="/api")

PLUGIN_DB_PATH = (
    Path.home() / ".claude" / "mcp-skill-hub" / "plugins" / "shadow_skill_evolution.db"
)
LOCAL_SKILLS_DIR = Path.home() / ".claude" / "local-skills"


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


@router.get("/proposals")
def list_proposals(status: str = "pending", limit: int = 50):
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


@router.post("/proposals/{proposal_id}/approve")
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

    return RedirectResponse(url="/skill-evolution?status=approved", status_code=303)


@router.post("/proposals/{proposal_id}/reject")
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

    return RedirectResponse(url="/skill-evolution?status=rejected", status_code=303)


@router.post("/analyze")
def trigger_analysis():
    """Trigger the propose_skills scheduled task manually."""
    import subprocess
    import sys

    script_path = (
        Path(__file__).resolve().parent.parent.parent
        / "scripts"
        / "propose_skills.py"
    )
    if script_path.exists():
        subprocess.run([sys.executable, str(script_path)], check=False)
        return JSONResponse({"status": "triggered"})
    return JSONResponse({"status": "error", "message": "Script not found"}, status_code=500)
