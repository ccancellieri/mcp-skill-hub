"""Vector route — SVG scatter via random projection.

Sources (toggle via ?source=skills|tasks|teachings|verdicts):
  skills    — embeddings table (skill_hub.db)
  tasks     — tasks.vector column (skill_hub.db)
  teachings — teachings.rule_vector (skill_hub.db)
  verdicts  — command_verdicts.vector (verdict_cache.db)
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from ... import vector_viz

router = APIRouter()

VERDICTS_DB = Path.home() / ".claude" / "mcp-skill-hub" / "command_verdicts.db"

SOURCES = ("skills", "tasks", "teachings", "verdicts")


def _load_skill_points(store: Any, limit: int) -> list[dict]:
    rows = store._conn.execute(
        "SELECT e.skill_id AS id, s.name AS label, s.target AS group_, "
        "       e.vector AS vec "
        "FROM embeddings e LEFT JOIN skills s ON s.id = e.skill_id "
        "LIMIT ?", (limit,),
    ).fetchall()
    out: list[dict] = []
    for r in rows:
        try:
            v = json.loads(r["vec"])
        except (TypeError, json.JSONDecodeError):
            continue
        out.append({
            "id": r["id"], "label": r["label"] or r["id"],
            "group": r["group_"] or "", "vector": v,
        })
    return out


def _load_task_points(store: Any, limit: int) -> list[dict]:
    try:
        rows = store._conn.execute(
            "SELECT id, title AS label, status AS group_, vector AS vec "
            "FROM tasks WHERE vector IS NOT NULL AND vector != '' LIMIT ?",
            (limit,),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    out: list[dict] = []
    for r in rows:
        try:
            v = json.loads(r["vec"])
        except (TypeError, json.JSONDecodeError):
            continue
        out.append({
            "id": str(r["id"]), "label": (r["label"] or "")[:60],
            "group": r["group_"] or "", "vector": v,
        })
    return out


def _load_teaching_points(store: Any, limit: int) -> list[dict]:
    try:
        rows = store._conn.execute(
            "SELECT id, rule AS label, action AS group_, "
            "       rule_vector AS vec FROM teachings LIMIT ?", (limit,),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    out: list[dict] = []
    for r in rows:
        try:
            v = json.loads(r["vec"])
        except (TypeError, json.JSONDecodeError):
            continue
        out.append({
            "id": str(r["id"]), "label": (r["label"] or "")[:60],
            "group": r["group_"] or "", "vector": v,
        })
    return out


def _load_verdict_points(limit: int) -> list[dict]:
    if not VERDICTS_DB.exists():
        return []
    try:
        conn = sqlite3.connect(str(VERDICTS_DB))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT cmd_hash AS id, command AS label, source AS group_, "
            "       vector AS vec FROM command_verdicts "
            "WHERE vector IS NOT NULL AND vector != '' LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
    except sqlite3.Error:
        return []
    out: list[dict] = []
    for r in rows:
        try:
            v = json.loads(r["vec"])
        except (TypeError, json.JSONDecodeError):
            continue
        out.append({
            "id": r["id"], "label": (r["label"] or "")[:60],
            "group": r["group_"] or "", "vector": v,
        })
    return out


def _project(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    dim = max(len(r["vector"]) for r in rows)
    mat = vector_viz.get_projection(dim_in=dim)
    return vector_viz.project_all(rows, matrix=mat)


def _color_for(group: str) -> str:
    if not group:
        return "#888"
    palette = ["#2b7bd6", "#2b9d5e", "#d6732b", "#c0392b", "#8e44ad",
               "#16a085", "#d4a017", "#7f8c8d", "#2980b9", "#27ae60"]
    return palette[hash(group) % len(palette)]


def _build_svg(points: list[dict], width: int = 720, height: int = 480) -> str:
    if not points:
        return (f'<svg width="{width}" height="{height}" '
                f'xmlns="http://www.w3.org/2000/svg">'
                f'<text x="20" y="30" fill="#888">no vectors for this source</text>'
                f'</svg>')
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    span_x = (xmax - xmin) or 1.0
    span_y = (ymax - ymin) or 1.0
    pad = 16

    def sx(v: float) -> float:
        return pad + (v - xmin) / span_x * (width - 2 * pad)

    def sy(v: float) -> float:
        return pad + (v - ymin) / span_y * (height - 2 * pad)

    parts = [
        f'<svg width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="background:#0e141b;border:1px solid #233;border-radius:6px">'
    ]
    for p in points:
        c = _color_for(p.get("group", ""))
        label = (p.get("label") or "").replace("<", "&lt;").replace(">", "&gt;")
        gid = (p.get("group") or "").replace("<", "&lt;").replace(">", "&gt;")
        parts.append(
            f'<circle cx="{sx(p["x"]):.1f}" cy="{sy(p["y"]):.1f}" r="3.5" '
            f'fill="{c}" fill-opacity="0.75" stroke="#000" stroke-width="0.3">'
            f'<title>{label}\n[{gid}]</title></circle>'
        )
    parts.append("</svg>")
    return "".join(parts)


@router.get("/vector", response_class=HTMLResponse)
def vector_page(request: Request, source: str = "skills",
                limit: int = 1000) -> Any:
    if source not in SOURCES:
        source = "skills"
    store = request.app.state.store
    if source == "skills":
        rows = _load_skill_points(store, limit)
    elif source == "tasks":
        rows = _load_task_points(store, limit)
    elif source == "teachings":
        rows = _load_teaching_points(store, limit)
    else:
        rows = _load_verdict_points(limit)
    points = _project(rows)
    svg = _build_svg(points)
    groups = sorted({p.get("group", "") for p in points if p.get("group")})
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request, "vector.html",
        {
            "active_tab": "vector",
            "sources": SOURCES,
            "source": source,
            "limit": limit,
            "count": len(points),
            "groups": groups,
            "svg": svg,
        },
    )
