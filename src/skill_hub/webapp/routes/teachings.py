"""Teachings route — list, add, delete, weight, search."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

router = APIRouter()

ACTIONS = ["suggest", "skip", "warn"]
TARGET_TYPES = ["task", "skill", "command", "general"]


def _try_embed(text: str) -> list[float] | None:
    try:
        from ... import embeddings as _emb
        vec = _emb.embed(text)
        if vec and isinstance(vec, list):
            return vec
    except Exception:
        return None
    return None


class TeachingCreate(BaseModel):
    rule: str
    action: str = "suggest"
    target_type: str = "general"
    target_id: int | None = None
    weight: float = 1.0


class TeachingWeightPatch(BaseModel):
    weight: float


def _all_teachings(store) -> list[dict]:
    return [dict(r) for r in store.list_teachings()]


def _render_row(templates, request, row: dict) -> HTMLResponse:
    return templates.TemplateResponse(request, "_teaching_row.html", {"t": row})


@router.get("/teachings", response_class=HTMLResponse)
def teachings_page(request: Request) -> Any:
    store = request.app.state.store
    rows = _all_teachings(store)
    embed_ok = _try_embed("ping") is not None
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "teachings.html",
        {
            "rows": rows,
            "actions": ACTIONS,
            "target_types": TARGET_TYPES,
            "embed_ok": embed_ok,
            "active_tab": "teachings",
        },
    )


@router.post("/teachings/add", response_class=HTMLResponse)
async def teachings_add(request: Request) -> Any:
    form = await request.form()
    rule = str(form.get("rule", "")).strip()
    if not rule:
        return HTMLResponse(
            '<div class="status err">rule required</div>', status_code=400
        )
    action = str(form.get("action", "suggest"))
    target_type = str(form.get("target_type", "general"))
    tid_raw = str(form.get("target_id", "")).strip()
    target_id: str = tid_raw if tid_raw else ""
    try:
        weight = float(form.get("weight", "1.0") or 1.0)
    except ValueError:
        weight = 1.0
    vec = _try_embed(rule) or []
    store = request.app.state.store
    warn = ""
    if not vec:
        warn = " (warning: embeddings unavailable — stored with empty vector)"
    new_id = store.add_teaching(
        rule=rule, rule_vector=vec, action=action,
        target_type=target_type, target_id=target_id, weight=weight,
    )
    # re-render table body
    rows = _all_teachings(store)
    templates = request.app.state.templates
    html = templates.TemplateResponse(
        request, "_teaching_rows.html", {"rows": rows},
    ).body.decode()
    return HTMLResponse(html + (
        f'<div class="status ok" hx-swap-oob="true" id="teach-status">'
        f'added #{new_id}{warn}</div>' if warn else ""
    ))


@router.post("/teachings/{teaching_id}/delete", response_class=HTMLResponse)
def teachings_delete(teaching_id: int, request: Request) -> HTMLResponse:
    store = request.app.state.store
    store.remove_teaching(teaching_id)
    return HTMLResponse("")


@router.patch("/teachings/{teaching_id}/weight", response_class=HTMLResponse)
async def teachings_weight(teaching_id: int, request: Request) -> Any:
    form = await request.form()
    try:
        w = float(form.get("weight", "1.0") or 1.0)
    except ValueError:
        return HTMLResponse(
            '<span class="status err">bad weight</span>', status_code=400
        )
    store = request.app.state.store
    store._conn.execute(
        "UPDATE teachings SET weight = ? WHERE id = ?", (w, teaching_id)
    )
    store._conn.commit()
    row = store._conn.execute(
        "SELECT id, rule, action, target_type, target_id, weight "
        "FROM teachings WHERE id = ?", (teaching_id,),
    ).fetchone()
    if not row:
        return HTMLResponse("", status_code=404)
    return _render_row(request.app.state.templates, request, dict(row))


@router.get("/teachings/search", response_class=HTMLResponse)
def teachings_search(request: Request, q: str = "") -> Any:
    store = request.app.state.store
    rows: list[dict] = []
    if q:
        vec = _try_embed(q)
        if vec:
            try:
                rows = [dict(r) for r in store.search_teachings(vec, min_sim=0.4)]
            except Exception:
                rows = []
        if not rows:
            # LIKE fallback
            like_rows = store._conn.execute(
                "SELECT id, rule, action, target_type, target_id, weight "
                "FROM teachings WHERE rule LIKE ? ORDER BY created_at DESC",
                (f"%{q}%",),
            ).fetchall()
            rows = [dict(r) for r in like_rows]
    else:
        rows = _all_teachings(store)
    templates = request.app.state.templates
    return templates.TemplateResponse(request, "_teaching_rows.html", {"rows": rows})
