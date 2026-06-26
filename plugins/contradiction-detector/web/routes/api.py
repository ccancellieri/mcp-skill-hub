"""API routes for contradiction resolution."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api")

DB_PATH = Path.home() / ".claude" / "mcp-skill-hub" / "skill_hub.db"

RESOLUTION_OPTIONS = [
    ("keep_a", "Keep A — Page A's claim is correct"),
    ("keep_b", "Keep B — Page B's claim is correct"),
    ("merge", "Merge — Combine both claims"),
    ("both_valid", "Both Valid — Different contexts, no conflict"),
]


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@router.post("/resolve")
async def resolve_finding(
    request: Request,
    finding_id: int = Form(...),
    resolution: str = Form(...),
    notes: str = Form(default=""),
):
    valid_resolutions = [r[0] for r in RESOLUTION_OPTIONS]
    if resolution not in valid_resolutions:
        return JSONResponse(
            {"error": f"Invalid resolution. Must be one of: {valid_resolutions}"},
            status_code=400,
        )

    with _get_conn() as conn:
        row = conn.execute(
            "SELECT id, page_a, page_b, claim_a, claim_b FROM plugin_contradiction_findings WHERE id = ?",
            (finding_id,),
        ).fetchone()

        if not row:
            return JSONResponse({"error": "Finding not found"}, status_code=404)

        resolved_at = datetime.utcnow().isoformat()
        conn.execute(
            """
            UPDATE plugin_contradiction_findings
            SET resolution_status = 'resolved',
                resolution = ?,
                resolved_by = 'user',
                resolved_at = ?
            WHERE id = ?
            """,
            (json.dumps({"action": resolution, "notes": notes}), resolved_at, finding_id),
        )
        conn.commit()

    return JSONResponse({
        "ok": True,
        "finding_id": finding_id,
        "resolution": resolution,
        "resolved_at": resolved_at,
    })


@router.post("/run-detection")
async def run_detection(request: Request, dry_run: bool = False):
    try:
        from scripts.detect_contradictions import detect_contradictions
        with _get_conn() as conn:
            result = detect_contradictions(conn, dry_run=dry_run)
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.get("/stats")
def get_stats(request: Request):
    with _get_conn() as conn:
        pending = conn.execute(
            "SELECT COUNT(*) FROM plugin_contradiction_findings WHERE resolution_status = 'pending'"
        ).fetchone()[0]

        resolved = conn.execute(
            "SELECT COUNT(*) FROM plugin_contradiction_findings WHERE resolution_status = 'resolved'"
        ).fetchone()[0]

        last_run = conn.execute(
            """
            SELECT id, started_at, completed_at, status, contradictions_found
            FROM plugin_contradiction_runs
            ORDER BY started_at DESC LIMIT 1
            """
        ).fetchone()

    return JSONResponse({
        "pending": pending,
        "resolved": resolved,
        "last_run": dict(last_run) if last_run else None,
    })


@router.get("/contradictions")
def list_findings(request: Request, status: str = "pending", limit: int = 50):
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, page_a, page_b, claim_a, claim_b, confidence, detected_at
            FROM plugin_contradiction_findings
            WHERE resolution_status = ?
            ORDER BY confidence DESC
            LIMIT ?
            """,
            (status, limit),
        ).fetchall()
    return JSONResponse([dict(r) for r in rows])
