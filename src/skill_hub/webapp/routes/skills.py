"""Skills route — usage stats, filter, details drawer, pin."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()

PIN_FILE = Path.home() / ".claude" / "mcp-skill-hub" / "state" / "pinned_skills.json"


def _load_pinned() -> set[str]:
    try:
        data = json.loads(PIN_FILE.read_text())
        if isinstance(data, list):
            return {str(x) for x in data}
    except (OSError, json.JSONDecodeError):
        pass
    return set()


def _save_pinned(pinned: set[str]) -> None:
    PIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    PIN_FILE.write_text(json.dumps(sorted(pinned)))


def _enrich(stats: list[dict], pinned: set[str]) -> list[dict]:
    out = []
    for s in stats:
        d = dict(s)
        d["pinned"] = d["id"] in pinned
        out.append(d)
    # pinned first, preserve stat ordering within each group
    out.sort(key=lambda d: (not d["pinned"],))
    return out


@router.get("/skills", response_class=HTMLResponse)
def skills_page(request: Request) -> Any:
    store = request.app.state.store
    stats = store.get_skill_usage_stats()
    pinned = _load_pinned()
    rows = _enrich(stats, pinned)
    # distinct targets / plugins for filter dropdowns
    targets: list[str] = sorted({(r.get("target") or "") for r in rows if r.get("target")})
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "skills.html",
        {
            "rows": rows,
            "targets": targets,
            "active_tab": "skills",
        },
    )


@router.post("/skills/{skill_id}/pin", response_class=HTMLResponse)
def skill_pin(skill_id: str, request: Request) -> Any:
    pinned = _load_pinned()
    if skill_id in pinned:
        pinned.discard(skill_id)
    else:
        pinned.add(skill_id)
    _save_pinned(pinned)
    store = request.app.state.store
    stats = store.get_skill_usage_stats()
    row = next((dict(r) for r in stats if r["id"] == skill_id), None)
    if not row:
        return HTMLResponse("", status_code=404)
    row["pinned"] = skill_id in pinned
    templates = request.app.state.templates
    return templates.TemplateResponse(request, "_skill_row.html", {"r": row})


@router.get("/skills/{skill_id}/detail", response_class=HTMLResponse)
def skill_detail(skill_id: str, request: Request) -> Any:
    store = request.app.state.store
    skill = store.get_skill(skill_id)
    if not skill:
        return HTMLResponse("<div class='muted'>Not found</div>", status_code=404)
    # Recent feedback rows for this skill
    try:
        fb_rows = store._conn.execute(
            "SELECT query, helpful, created_at FROM feedback "
            "WHERE skill_id = ? ORDER BY created_at DESC LIMIT 15",
            (skill_id,),
        ).fetchall()
        feedback = [dict(r) for r in fb_rows]
    except Exception:
        feedback = []
    # Embedding vector norm
    norm: float | None = None
    try:
        vrow = store._conn.execute(
            "SELECT norm, vector FROM embeddings WHERE skill_id = ?",
            (skill_id,),
        ).fetchone()
        if vrow:
            if vrow["norm"] is not None:
                norm = float(vrow["norm"])
            elif vrow["vector"]:
                vec = json.loads(vrow["vector"])
                norm = math.sqrt(sum(x * x for x in vec))
    except Exception:
        norm = None
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "_skill_detail.html",
        {"s": skill, "feedback": feedback, "norm": norm},
    )
