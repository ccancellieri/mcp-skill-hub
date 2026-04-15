"""Vector route — SVG scatter via random projection + index catalog.

Sources (toggle via ?source=skills|tasks|teachings|verdicts|namespaces):
  skills     — embeddings table (skill_hub.db)
  tasks      — tasks.vector column (skill_hub.db)
  teachings  — teachings.rule_vector (skill_hub.db)
  verdicts   — command_verdicts.vector (verdict_cache.db)
  namespaces — vectors table filtered by namespace + level
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

SOURCES = ("skills", "tasks", "teachings", "verdicts", "namespaces")
ALL_LEVELS = ("L0", "L1", "L2", "L3", "L4")

# ── point loaders ─────────────────────────────────────────────────────────────

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


def _load_namespace_points(
    store: Any, namespace: str, levels: list[str], limit: int
) -> list[dict]:
    """Load points from the unified vectors table, filtered by namespace + levels."""
    try:
        if levels:
            placeholders = ",".join("?" * len(levels))
            rows = store._conn.execute(
                f"SELECT doc_id AS id, metadata AS meta, namespace AS group_, "
                f"       level, vector AS vec "
                f"FROM vectors WHERE namespace = ? AND level IN ({placeholders}) "
                f"AND vec IS NOT NULL LIMIT ?",
                [namespace, *levels, limit],
            ).fetchall()
        else:
            rows = store._conn.execute(
                "SELECT doc_id AS id, metadata AS meta, namespace AS group_, "
                "       level, vector AS vec "
                "FROM vectors WHERE namespace = ? AND vec IS NOT NULL LIMIT ?",
                (namespace, limit),
            ).fetchall()
    except sqlite3.OperationalError:
        return []
    out: list[dict] = []
    for r in rows:
        try:
            v = json.loads(r["vec"])
        except (TypeError, json.JSONDecodeError):
            continue
        label = r["id"]
        try:
            meta = json.loads(r["meta"] or "{}")
            label = meta.get("title") or meta.get("name") or label
        except (TypeError, json.JSONDecodeError):
            pass
        out.append({
            "id": str(r["id"]), "label": str(label)[:60],
            "group": r["level"] or r["group_"] or "", "vector": v,
        })
    return out


# ── projection + SVG ─────────────────────────────────────────────────────────

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


def _build_svg(points: list[dict], width: int = 720, height: int = 440) -> str:
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
        f'style="background:#0e141b;border:1px solid #233;border-radius:6px;width:100%;height:auto">'
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


# ── index catalog ─────────────────────────────────────────────────────────────

def _load_index_catalog(store: Any) -> list[dict]:
    """Return all vector_index_config rows enriched with live doc counts."""
    try:
        configs = store._conn.execute(
            "SELECT name, embedding_model, chunk_size, chunk_overlap, "
            "       default_level, half_life_days, max_docs, updated_at "
            "FROM vector_index_config ORDER BY name"
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    # Aggregate doc counts from vectors table (may not exist yet)
    counts: dict[str, dict] = {}
    try:
        rows = store._conn.execute(
            "SELECT namespace, COUNT(*) AS n, "
            "       ROUND(AVG((julianday('now') - julianday(indexed_at))), 1) AS avg_age, "
            "       SUM(access_count) AS total_access, "
            "       MAX(indexed_at) AS last_indexed "
            "FROM vectors GROUP BY namespace"
        ).fetchall()
        for r in rows:
            counts[r["namespace"]] = {
                "doc_count": r["n"],
                "avg_age_days": r["avg_age"],
                "total_access": r["total_access"],
                "last_indexed": (r["last_indexed"] or "")[:16],
            }
    except sqlite3.OperationalError:
        pass

    # Aggregate per-level breakdown
    level_counts: dict[str, dict[str, int]] = {}
    try:
        rows = store._conn.execute(
            "SELECT namespace, level, COUNT(*) AS n FROM vectors GROUP BY namespace, level"
        ).fetchall()
        for r in rows:
            ns = r["namespace"]
            if ns not in level_counts:
                level_counts[ns] = {}
            level_counts[ns][r["level"] or "?"] = r["n"]
    except sqlite3.OperationalError:
        pass

    result = []
    for c in configs:
        ns = c["name"]
        stats = counts.get(ns, {})
        lvl_breakdown = level_counts.get(ns, {})
        lvl_str = "  ".join(
            f"{lv}:{n}" for lv, n in sorted(lvl_breakdown.items())
        ) if lvl_breakdown else "—"
        result.append({
            "name": ns,
            "default_level": c["default_level"] or "L2",
            "half_life_days": c["half_life_days"],
            "chunk_size": c["chunk_size"] or 0,
            "chunk_overlap": c["chunk_overlap"] or 0,
            "max_docs": c["max_docs"] or 0,
            "embedding_model": c["embedding_model"] or "—",
            "doc_count": stats.get("doc_count", 0),
            "avg_age_days": stats.get("avg_age_days", "—"),
            "total_access": stats.get("total_access", 0),
            "last_indexed": stats.get("last_indexed", "—"),
            "level_breakdown": lvl_str,
            "updated_at": (c["updated_at"] or "")[:16],
        })
    return result


def _load_namespaces(store: Any) -> list[str]:
    """Return all known namespaces (union of config + live vectors)."""
    names: set[str] = set()
    try:
        for r in store._conn.execute("SELECT name FROM vector_index_config").fetchall():
            names.add(r["name"])
    except sqlite3.OperationalError:
        pass
    try:
        for r in store._conn.execute("SELECT DISTINCT namespace FROM vectors").fetchall():
            names.add(r["namespace"])
    except sqlite3.OperationalError:
        pass
    return sorted(names)


# ── route ─────────────────────────────────────────────────────────────────────

@router.get("/vector", response_class=HTMLResponse)
def vector_page(
    request: Request,
    source: str = "skills",
    limit: int = 1000,
    namespace: str = "",
    levels: str = "",          # comma-separated e.g. "L1,L2,L3"
    tab: str = "scatter",      # scatter | indexes
) -> Any:
    if source not in SOURCES:
        source = "skills"
    store = request.app.state.store

    # Parse level filter
    selected_levels = [l.strip() for l in levels.split(",") if l.strip() in ALL_LEVELS]

    # Load scatter points
    if source == "skills":
        rows = _load_skill_points(store, limit)
    elif source == "tasks":
        rows = _load_task_points(store, limit)
    elif source == "teachings":
        rows = _load_teaching_points(store, limit)
    elif source == "verdicts":
        rows = _load_verdict_points(limit)
    else:  # namespaces
        ns = namespace or "skills"
        rows = _load_namespace_points(store, ns, selected_levels, limit)

    points = _project(rows)
    svg = _build_svg(points)
    groups = sorted({p.get("group", "") for p in points if p.get("group")})

    # Load index catalog
    catalog = _load_index_catalog(store)
    all_namespaces = _load_namespaces(store)

    # Color map for legend (consistent colors)
    legend = [
        {"group": g, "color": _color_for(g)}
        for g in groups
    ]

    templates = request.app.state.templates
    return templates.TemplateResponse(
        request, "vector.html",
        {
            "active_tab": "vector",
            "tab": tab,
            "sources": SOURCES,
            "source": source,
            "limit": limit,
            "sel_namespace": namespace or (all_namespaces[0] if all_namespaces else ""),
            "all_namespaces": all_namespaces,
            "selected_levels": selected_levels,
            "all_levels": ALL_LEVELS,
            "count": len(points),
            "groups": groups,
            "legend": legend,
            "svg": svg,
            "catalog": catalog,
        },
    )
