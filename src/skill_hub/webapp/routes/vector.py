"""Vector route — interactive scatter + index catalog + LLM analysis.

Endpoints:
  GET  /vector              — main page (HTML)
  GET  /vector/points       — projected points as JSON
  POST /vector/analyze      — LLM analysis of a selection
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ... import vector_viz

router = APIRouter()

VERDICTS_DB = Path.home() / ".claude" / "mcp-skill-hub" / "command_verdicts.db"

SOURCES = ("skills", "tasks", "teachings", "verdicts", "namespaces")
ALL_LEVELS = ("L0", "L1", "L2", "L3", "L4")


# ── point loaders (return enriched dicts) ────────────────────────────────────

def _load_skill_points(store: Any, limit: int) -> list[dict]:
    rows = store._conn.execute(
        "SELECT e.skill_id AS id, s.name AS label, s.target AS group_, "
        "       s.description, s.plugin, s.feedback_score, s.file_path, "
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
            "id": r["id"],
            "label": r["label"] or r["id"],
            "group": r["group_"] or "",
            "vector": v,
            "url": f"/skills?focus={r['id']}",
            "meta": {
                "type": "skill",
                "target": r["group_"] or "",
                "plugin": r["plugin"] or "",
                "description": (r["description"] or "")[:200],
                "feedback_score": r["feedback_score"] or 0,
            },
        })
    return out


def _load_task_points(store: Any, limit: int) -> list[dict]:
    try:
        rows = store._conn.execute(
            "SELECT id, title AS label, status AS group_, summary, "
            "       created_at, tags, vector AS vec "
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
            "id": str(r["id"]),
            "label": (r["label"] or "")[:60],
            "group": r["group_"] or "",
            "vector": v,
            "url": f"/tasks?id={r['id']}",
            "meta": {
                "type": "task",
                "status": r["group_"] or "",
                "created_at": (r["created_at"] or "")[:10],
                "summary": (r["summary"] or "")[:200],
                "tags": r["tags"] or "",
            },
        })
    return out


def _load_teaching_points(store: Any, limit: int) -> list[dict]:
    try:
        rows = store._conn.execute(
            "SELECT id, rule AS label, action AS group_, weight, "
            "       target_type, created_at, rule_vector AS vec "
            "FROM teachings LIMIT ?", (limit,),
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
            "id": str(r["id"]),
            "label": (r["label"] or "")[:60],
            "group": r["group_"] or "",
            "vector": v,
            "url": "/teachings",
            "meta": {
                "type": "teaching",
                "action": r["group_"] or "",
                "weight": r["weight"] or 1.0,
                "target_type": r["target_type"] or "",
                "created_at": (r["created_at"] or "")[:10],
            },
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
            "       verdict, vector AS vec FROM command_verdicts "
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
            "id": r["id"],
            "label": (r["label"] or "")[:60],
            "group": r["group_"] or "",
            "vector": v,
            "url": "/verdicts",
            "meta": {
                "type": "verdict",
                "source": r["group_"] or "",
                "verdict": r["verdict"] or "",
            },
        })
    return out


def _load_namespace_points(
    store: Any, namespace: str, levels: list[str], limit: int
) -> list[dict]:
    try:
        if levels:
            placeholders = ",".join("?" * len(levels))
            rows = store._conn.execute(
                f"SELECT doc_id AS id, metadata AS meta_json, namespace, "
                f"       level, access_count, last_accessed, vector AS vec "
                f"FROM vectors WHERE namespace = ? AND level IN ({placeholders}) "
                f"AND vec IS NOT NULL LIMIT ?",
                [namespace, *levels, limit],
            ).fetchall()
        else:
            rows = store._conn.execute(
                "SELECT doc_id AS id, metadata AS meta_json, namespace, "
                "       level, access_count, last_accessed, vector AS vec "
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
        extra: dict = {}
        try:
            m = json.loads(r["meta_json"] or "{}")
            label = m.get("title") or m.get("name") or label
            extra = m
        except (TypeError, json.JSONDecodeError):
            pass
        out.append({
            "id": str(r["id"]),
            "label": str(label)[:60],
            "group": r["level"] or "",
            "vector": v,
            "url": f"/vector?source=namespaces&namespace={namespace}&tab=scatter",
            "meta": {
                "type": "memory",
                "namespace": r["namespace"],
                "level": r["level"] or "L2",
                "access_count": r["access_count"] or 0,
                "last_accessed": (r["last_accessed"] or "")[:10],
                **{k: str(v2)[:120] for k, v2 in extra.items()
                   if isinstance(v2, (str, int, float))},
            },
        })
    return out


# ── projection helpers ────────────────────────────────────────────────────────

def _project(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    dim = max(len(r["vector"]) for r in rows)
    mat = vector_viz.get_projection(dim_in=dim)
    projected = vector_viz.project_all(rows, matrix=mat)
    # merge back url + meta fields by id (project_all only returns id/x/y/label/group)
    extra: dict[str, dict] = {str(r["id"]): r for r in rows}
    for p in projected:
        orig = extra.get(str(p.get("id")), {})
        p["url"] = orig.get("url", "")
        p["meta"] = orig.get("meta", {})
    return projected


def _color_for(group: str) -> str:
    if not group:
        return "#888"
    palette = ["#2b7bd6", "#2b9d5e", "#d6732b", "#c0392b", "#8e44ad",
               "#16a085", "#d4a017", "#7f8c8d", "#2980b9", "#27ae60"]
    return palette[hash(group) % len(palette)]


# ── index catalog ─────────────────────────────────────────────────────────────

def _load_index_catalog(registry: Any) -> list[dict]:
    """Render every MergeableSource as a catalog row."""
    out: list[dict] = []
    for src in registry.all():
        s = src.index_stats()
        lvl_str = "  ".join(
            f"{lv}:{n}" for lv, n in sorted(s.level_breakdown.items())
        ) or "—"
        out.append({
            "name": s.name,
            "source_type": s.source_type,
            "doc_count": s.doc_count,
            "embedded_count": s.embedded_count,
            "embedding_model": s.embedding_model or "—",
            "last_indexed": s.last_indexed or "—",
            "level_breakdown": lvl_str,
            "scatter_url": s.scatter_url,
            "supports_merge": s.supports_merge,
            "merge_mode": s.merge_mode.value,
        })
    return out


def _load_namespaces(store: Any) -> list[str]:
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


# ── full-content loaders for LLM analysis ────────────────────────────────────

def _fetch_skill_content(store: Any, ids: list[str]) -> list[dict]:
    if not ids:
        return []
    ph = ",".join("?" * len(ids))
    try:
        rows = store._conn.execute(
            f"SELECT id, name, description, content, target, plugin "
            f"FROM skills WHERE id IN ({ph})", ids,
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [{"id": r["id"], "label": r["name"] or r["id"],
             "body": f"[{r['target']}] {r['description'] or ''}\n{(r['content'] or '')[:1000]}"
             } for r in rows]


def _fetch_task_content(store: Any, ids: list[str]) -> list[dict]:
    if not ids:
        return []
    ph = ",".join("?" * len(ids))
    try:
        rows = store._conn.execute(
            f"SELECT id, title, summary, status, context FROM tasks WHERE id IN ({ph})",
            [int(i) for i in ids if i.isdigit()],
        ).fetchall()
    except (sqlite3.OperationalError, ValueError):
        return []
    return [{"id": str(r["id"]), "label": r["title"] or f"task:{r['id']}",
             "body": f"[{r['status']}] {r['summary'] or ''}\n{(r['context'] or '')[:500]}"
             } for r in rows]


def _fetch_teaching_content(store: Any, ids: list[str]) -> list[dict]:
    if not ids:
        return []
    ph = ",".join("?" * len(ids))
    try:
        rows = store._conn.execute(
            f"SELECT id, rule, action, target_type, weight FROM teachings WHERE id IN ({ph})",
            [int(i) for i in ids if i.isdigit()],
        ).fetchall()
    except (sqlite3.OperationalError, ValueError):
        return []
    return [{"id": str(r["id"]), "label": (r["rule"] or "")[:60],
             "body": f"[{r['action']}] rule: {r['rule'] or ''} (weight={r['weight']}, type={r['target_type'] or ''})"
             } for r in rows]


def _fetch_namespace_content(store: Any, ids: list[str], namespace: str) -> list[dict]:
    if not ids:
        return []
    ph = ",".join("?" * len(ids))
    try:
        rows = store._conn.execute(
            f"SELECT doc_id, metadata, level, access_count "
            f"FROM vectors WHERE namespace = ? AND doc_id IN ({ph})",
            [namespace, *ids],
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    out = []
    for r in rows:
        meta: dict = {}
        try:
            meta = json.loads(r["metadata"] or "{}")
        except (TypeError, json.JSONDecodeError):
            pass
        label = meta.get("title") or meta.get("name") or r["doc_id"]
        body = meta.get("content") or meta.get("text") or meta.get("body") or str(meta)[:500]
        out.append({"id": r["doc_id"], "label": str(label)[:60],
                    "body": f"[{r['level']}] {body[:800]}"})
    return out


# ── routes ────────────────────────────────────────────────────────────────────

def _build_points(
    store: Any,
    source: str,
    limit: int,
    namespace: str,
    selected_levels: list[str],
) -> list[dict]:
    if source == "skills":
        rows = _load_skill_points(store, limit)
    elif source == "tasks":
        rows = _load_task_points(store, limit)
    elif source == "teachings":
        rows = _load_teaching_points(store, limit)
    elif source == "verdicts":
        rows = _load_verdict_points(limit)
    else:
        ns = namespace or "skills"
        rows = _load_namespace_points(store, ns, selected_levels, limit)
    return _project(rows)


@router.get("/vector", response_class=HTMLResponse)
def vector_page(
    request: Request,
    source: str = "skills",
    limit: int = 1000,
    namespace: str = "",
    levels: str = "",
    tab: str = "scatter",
) -> Any:
    if source not in SOURCES:
        source = "skills"
    store = request.app.state.store
    selected_levels = [l.strip() for l in levels.split(",") if l.strip() in ALL_LEVELS]

    points = _build_points(store, source, limit, namespace, selected_levels)
    groups = sorted({p.get("group", "") for p in points if p.get("group")})
    legend = [{"group": g, "color": _color_for(g)} for g in groups]

    registry = request.app.state.source_registry
    catalog = _load_index_catalog(registry)
    all_namespaces = _load_namespaces(store)

    # colour map for JS: {group -> hex}
    color_map = {g: _color_for(g) for g in groups}
    color_map[""] = "#888"

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
            "catalog": catalog,
            # serialised for JS
            # Escape `</` to guard against `</script>` breakouts inside the
            # <script type="application/json"> data island (template uses |safe).
            "points_json": json.dumps(points).replace("</", "<\\/"),
            "color_map_json": json.dumps(color_map).replace("</", "<\\/"),
        },
    )


@router.get("/vector/points", response_class=JSONResponse)
def vector_points(
    request: Request,
    source: str = "skills",
    limit: int = 1000,
    namespace: str = "",
    levels: str = "",
) -> Any:
    if source not in SOURCES:
        source = "skills"
    store = request.app.state.store
    selected_levels = [l.strip() for l in levels.split(",") if l.strip() in ALL_LEVELS]
    points = _build_points(store, source, limit, namespace, selected_levels)
    color_map = {p.get("group", ""): _color_for(p.get("group", "")) for p in points}
    color_map[""] = "#888"
    return {"points": points, "color_map": color_map, "count": len(points)}


from ...vector_sources import MergeDraft, MergeMode


@router.post("/vector/merge/draft", response_class=JSONResponse)
async def vector_merge_draft(request: Request) -> Any:
    body = await request.json()
    source_name: str = body.get("source", "")
    ids: list[str] = body.get("ids", [])
    tier: str = body.get("tier", "local")
    instruction: str = body.get("instruction", "")
    # In-session directive controls (no API key required server-side):
    #   model: ui label "haiku|sonnet|opus" — picks the subagent model
    #   operations: ["consolidate", "promote", ...] — woven into the prompt
    model: str | None = body.get("model")
    operations: list[str] | None = body.get("operations")
    if not isinstance(operations, list):
        operations = None

    if len(ids) < 2:
        return JSONResponse({"error": "merge requires at least 2 items"}, status_code=400)

    registry = request.app.state.source_registry
    try:
        src = registry.get(source_name)
    except KeyError:
        return JSONResponse({"error": f"unknown source: {source_name}"}, status_code=400)

    if src.merge_mode == MergeMode.REJECTED:
        return JSONResponse(
            {"error": f"merge_rejected_for_source: {source_name}"}, status_code=400,
        )

    items = src.fetch_for_merge(ids)
    if len(items) != len(ids):
        missing = set(ids) - {it.id for it in items}
        return JSONResponse(
            {"error": "missing_ids", "missing": sorted(missing)}, status_code=404,
        )

    try:
        # Sources accept model/operations as kwargs. Older Task-style sources
        # ignore them (mechanical-only); LLM-mode sources weave them into the
        # directive. We pass via try/except to remain compatible with sources
        # that don't take the new kwargs yet.
        try:
            draft = src.draft_merge(
                items, tier=tier, instruction=instruction,
                model=model, operations=operations,
            )
        except TypeError:
            draft = src.draft_merge(items, tier=tier, instruction=instruction)
    except Exception as exc:
        return JSONResponse({"error": f"draft_failed: {exc}"}, status_code=503)

    return {
        "proposed_label": draft.proposed_label,
        "proposed_body": draft.proposed_body,
        "proposed_raw": draft.proposed_raw,
        "tier_used": draft.tier_used,
        "tokens_used": draft.tokens_used,
        "directive": draft.directive,
        "source": source_name,
        "ids": ids,
    }


@router.post("/vector/merge/commit", response_class=JSONResponse)
async def vector_merge_commit(request: Request) -> Any:
    body = await request.json()
    source_name: str = body.get("source", "")
    ids: list[str] = body.get("ids", [])
    draft_raw: dict = body.get("draft") or {}

    if len(ids) < 2:
        return JSONResponse({"error": "merge requires at least 2 items"}, status_code=400)

    registry = request.app.state.source_registry
    try:
        src = registry.get(source_name)
    except KeyError:
        return JSONResponse({"error": f"unknown source: {source_name}"}, status_code=400)

    if src.merge_mode == MergeMode.REJECTED:
        return JSONResponse(
            {"error": f"merge_rejected_for_source: {source_name}"}, status_code=400,
        )

    items = src.fetch_for_merge(ids)
    if len(items) != len(ids):
        missing = set(ids) - {it.id for it in items}
        return JSONResponse(
            {"error": "source_changed", "missing": sorted(missing)}, status_code=409,
        )

    try:
        draft = MergeDraft(
            proposed_label=draft_raw.get("proposed_label", ""),
            proposed_body=draft_raw.get("proposed_body", ""),
            proposed_raw=draft_raw.get("proposed_raw", {}),
            tier_used=draft_raw.get("tier_used", "mechanical"),
            tokens_used=draft_raw.get("tokens_used"),
            directive=draft_raw.get("directive"),
        )
        result = src.commit_merge(items, draft)
    except Exception as exc:
        return JSONResponse({"error": f"commit_failed: {exc}"}, status_code=500)

    return {
        "new_id": result.new_id,
        "closed_ids": result.closed_ids,
        "audit_id": result.audit_id,
    }


@router.post("/vector/merge", response_class=JSONResponse)
async def vector_merge_legacy(request: Request) -> Any:
    """Backward-compat shim: drives TaskSource mechanical draft+commit."""
    body = await request.json()
    source_name: str = body.get("source", "tasks")
    ids: list[str] = body.get("ids", [])
    if source_name != "tasks" or len(ids) < 2:
        return JSONResponse(
            {"error": "legacy shim supports tasks with 2+ ids only"}, status_code=400,
        )

    registry = request.app.state.source_registry
    src = registry.get("tasks")
    items = src.fetch_for_merge(ids)
    draft = src.draft_merge(items, tier="mechanical", instruction="")
    result = src.commit_merge(items, draft)
    return {"merged_into_id": int(result.new_id) if result.new_id else 0}


@router.post("/vector/analyze", response_class=JSONResponse)
async def vector_analyze(request: Request) -> Any:
    body = await request.json()
    ids: list[str] = body.get("ids", [])
    source: str = body.get("source", "skills")
    namespace: str = body.get("namespace", "")
    model_key: str = body.get("model", "local")   # "local" | "mid" | "claude" | explicit
    instruction: str = body.get("instruction", "").strip()

    if not ids:
        return JSONResponse({"error": "no ids provided"}, status_code=400)
    if not instruction:
        instruction = "Analyze these memory items. Identify patterns, redundancies, and suggest improvements."

    store = request.app.state.store

    # Load full content
    if source == "skills":
        items = _fetch_skill_content(store, ids)
    elif source == "tasks":
        items = _fetch_task_content(store, ids)
    elif source == "teachings":
        items = _fetch_teaching_content(store, ids)
    elif source == "namespaces":
        items = _fetch_namespace_content(store, ids, namespace or "skills")
    else:
        items = [{"id": i, "label": i, "body": ""} for i in ids]

    if not items:
        return JSONResponse({"error": "could not load content for selected items"}, status_code=404)

    # Build prompt
    items_text = "\n\n".join(
        f"### [{i+1}] {it['label']} (id={it['id']})\n{it['body']}"
        for i, it in enumerate(items[:30])  # cap at 30
    )
    prompt = (
        f"You are analyzing a selection of {source} items from a knowledge store.\n\n"
        f"INSTRUCTION: {instruction}\n\n"
        f"ITEMS ({len(items)} selected):\n\n{items_text}\n\n"
        f"Provide concise, actionable insights. Use markdown."
    )

    # Resolve model
    tier_map = {
        "local": "tier_cheap",
        "mid": "tier_mid",
        "claude": "tier_strong",
    }
    tier = tier_map.get(model_key, "tier_cheap")
    explicit_model: str | None = model_key if model_key not in tier_map else None

    try:
        from ...llm import get_provider
        provider = get_provider()
        result = provider.complete(
            prompt,
            tier=tier,
            model=explicit_model,
            max_tokens=1500,
            temperature=0.3,
            timeout=120.0,
        )
        return {"result": result, "model": model_key, "item_count": len(items)}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
