"""Tasks route — open/closed panels, CRUD, search, teaching, auto-approve."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

router = APIRouter()


def _try_embed(text: str) -> list[float] | None:
    """Return embedding vector or None if embeddings unavailable."""
    try:
        from ... import embeddings as _emb
        vec = _emb.embed(text)
        if vec and isinstance(vec, list):
            return vec
    except (ImportError, ConnectionError, OSError, Exception):
        return None
    return None


def _row_to_dict(row) -> dict:
    if row is None:
        return {}
    d = dict(row)
    return d


def _list_for_panel(store, status: str, limit: int | None = None) -> list[dict]:
    rows = store.list_tasks(status=status)
    out = []
    for r in rows:
        d = dict(r)
        # attach auto_approve flag (best-effort; method doesn't throw)
        try:
            d["auto_approve"] = store.get_task_auto_approve(d["id"])
        except Exception:
            d["auto_approve"] = None
        out.append(d)
    if limit:
        out = out[:limit]
    return out


@router.get("/tasks", response_class=HTMLResponse)
def tasks_page(request: Request) -> Any:
    store = request.app.state.store
    open_tasks = _list_for_panel(store, "open")
    closed_tasks = _list_for_panel(store, "closed", limit=50)
    templates = request.app.state.templates
    embed_ok = _try_embed("ping") is not None
    return templates.TemplateResponse(
        request,
        "tasks.html",
        {
            "open_tasks": open_tasks,
            "closed_tasks": closed_tasks,
            "embed_ok": embed_ok,
            "active_tab": "tasks",
        },
    )


def _render_row(templates, request, task: dict, panel: str) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "_task_row.html", {"t": task, "panel": panel}
    )


@router.patch("/tasks/{task_id}/title", response_class=HTMLResponse)
async def rename_task(task_id: int, request: Request) -> Any:
    form = await request.form()
    title = str(form.get("title", "")).strip()
    if not title:
        return HTMLResponse("title required", status_code=400)
    store = request.app.state.store
    store.rename_task_title(task_id, title)
    row = store.get_task(task_id)
    d = dict(row)
    try:
        d["auto_approve"] = store.get_task_auto_approve(task_id)
    except Exception:
        d["auto_approve"] = None
    panel = "open" if d.get("status") == "open" else "closed"
    return _render_row(request.app.state.templates, request, d, panel)


@router.post("/tasks/{task_id}/update", response_class=HTMLResponse)
async def update_task(task_id: int, request: Request) -> Any:
    form = await request.form()
    summary = str(form.get("summary", ""))
    tags = str(form.get("tags", ""))
    store = request.app.state.store
    store.update_task(task_id, summary=summary, tags=tags)
    row = store.get_task(task_id)
    d = dict(row)
    try:
        d["auto_approve"] = store.get_task_auto_approve(task_id)
    except Exception:
        d["auto_approve"] = None
    panel = "open" if d.get("status") == "open" else "closed"
    return _render_row(request.app.state.templates, request, d, panel)


def _auto_compact(task: dict) -> str:
    """Generate a compaction summary: LLM if available, otherwise task title."""
    content = (task.get("summary") or task.get("context") or "").strip()
    title = (task.get("title") or "").strip()
    if not content:
        return title or "closed via dashboard"
    try:
        from ... import embeddings as _emb
        result = _emb.compact(content)
        if isinstance(result, dict):
            parts = [result.get("summary") or ""]
            decisions = result.get("decisions") or []
            if decisions:
                parts.append("Decisions: " + "; ".join(str(d) for d in decisions[:3]))
            return "\n".join(p for p in parts if p).strip() or title
    except Exception:
        pass
    return title or content[:200]


@router.post("/tasks/{task_id}/close", response_class=HTMLResponse)
async def close_task(task_id: int, request: Request) -> Any:
    form = await request.form()
    summary = str(form.get("summary", "")).strip()
    store = request.app.state.store
    if not summary:
        task = _row_to_dict(store.get_task(task_id))
        summary = _auto_compact(task)
    store.close_task(task_id, compact=summary, compact_vector=None)
    row = store.get_task(task_id)
    d = dict(row) if row else {}
    d["auto_approve"] = None
    return _render_row(request.app.state.templates, request, d, "closed")


@router.get("/tasks/{task_id}/suggest-summary")
def suggest_summary(task_id: int, request: Request) -> JSONResponse:
    """Return a suggested compaction summary (LLM or title fallback)."""
    store = request.app.state.store
    task = _row_to_dict(store.get_task(task_id))
    if not task:
        return JSONResponse({"summary": "", "source": "not_found"}, status_code=404)
    content = (task.get("summary") or task.get("context") or "").strip()
    title = (task.get("title") or "").strip()
    if content:
        try:
            from ... import embeddings as _emb
            result = _emb.compact(content)
            if isinstance(result, dict) and (result.get("summary") or result.get("title")):
                parts = []
                if result.get("title") and result["title"] != "Untitled":
                    parts.append(result["title"])
                if result.get("summary"):
                    parts.append(result["summary"])
                decisions = result.get("decisions") or []
                if decisions:
                    parts.append("Decisions: " + "; ".join(str(d) for d in decisions[:3]))
                text = "\n".join(p for p in parts if p).strip()
                if text:
                    return JSONResponse({"summary": text, "source": "llm"})
        except Exception:
            pass
    return JSONResponse({"summary": title, "source": "title"})


@router.post("/tasks/{task_id}/reopen", response_class=HTMLResponse)
def reopen_task(task_id: int, request: Request) -> Any:
    store = request.app.state.store
    store.reopen_task(task_id)
    row = store.get_task(task_id)
    d = dict(row) if row else {}
    try:
        d["auto_approve"] = store.get_task_auto_approve(task_id)
    except Exception:
        d["auto_approve"] = None
    return _render_row(request.app.state.templates, request, d, "open")


@router.post("/tasks/{task_id}/delete", response_class=HTMLResponse)
def delete_task(task_id: int, request: Request) -> HTMLResponse:
    store = request.app.state.store
    store.delete_task(task_id)
    return HTMLResponse("")


@router.post("/tasks/{task_id}/teach")
async def teach_from_task(task_id: int, request: Request) -> JSONResponse:
    form = await request.form()
    rule = str(form.get("rule", "")).strip()
    if not rule:
        return JSONResponse({"error": "rule required"}, status_code=400)
    store = request.app.state.store
    vec = _try_embed(rule) or []
    if not vec:
        return JSONResponse(
            {"error": "embeddings unavailable (Ollama offline?)"},
            status_code=503,
        )
    tid = store.add_teaching(
        rule=rule, rule_vector=vec, action="suggest",
        target_type="task", target_id=str(task_id),
    )
    return JSONResponse({"teaching_id": tid})


@router.post("/tasks/{task_id}/auto_approve", response_class=HTMLResponse)
def auto_approve(task_id: int, enabled: str, request: Request) -> Any:
    if enabled == "null":
        val: bool | None = None
    elif enabled == "true":
        val = True
    else:
        val = False
    store = request.app.state.store
    store.set_task_auto_approve(task_id, val)
    row = store.get_task(task_id)
    d = dict(row) if row else {}
    d["auto_approve"] = val
    panel = "open" if d.get("status") == "open" else "closed"
    return _render_row(request.app.state.templates, request, d, panel)


class MergeBody(BaseModel):
    ids: list[int]


@router.post("/tasks/merge")
def merge_tasks(body: MergeBody, request: Request) -> JSONResponse:
    store = request.app.state.store
    new_id = store.merge_tasks(body.ids)
    return JSONResponse({"new_task_id": new_id})


@router.get("/tasks/search", response_class=HTMLResponse)
def search_tasks(request: Request, q: str = "", mode: str = "text") -> Any:
    store = request.app.state.store
    results: list[dict] = []
    error = ""
    if q:
        if mode == "semantic":
            vec = _try_embed(q)
            if not vec:
                error = "embeddings unavailable"
            else:
                results = store.search_tasks(vec, top_k=20, status="all")
        else:
            rows = store.search_tasks_text(q, status="all")
            results = [dict(r) for r in rows]
    for d in results:
        try:
            d["auto_approve"] = store.get_task_auto_approve(d["id"])
        except Exception:
            d["auto_approve"] = None
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "_task_search.html",
        {"results": results, "q": q, "mode": mode, "error": error},
    )


def _parse_router_log(
    session_id: str,
    created_at: str | None = None,
    closed_at: str | None = None,
) -> dict:
    """Parse router.jsonl for model routing stats for a given task window.

    Strategy:
      1. Exact session_id match (fast path).
      2. Time-window fallback: entries whose timestamp falls between
         created_at and closed_at (or now for open tasks).
         Needed because the MCP server's internal session UUID differs from
         the Claude Code conversation UUID recorded in the router log.
    """
    import json as _json
    from pathlib import Path
    from collections import Counter
    from datetime import datetime

    _EMPTY: dict = {"models": [], "plan_mode_count": 0, "tier_counts": {},
                    "total_prompts": 0, "top_skills": [], "matched_by": "none"}

    router_log = Path.home() / ".claude" / "mcp-skill-hub" / "router.jsonl"
    if not router_log.exists():
        return _EMPTY.copy()

    # Parse task time window for fallback (stored as "YYYY-MM-DD HH:MM:SS")
    ts_start: str | None = None
    ts_end: str | None = None
    try:
        if created_at:
            ts_start = created_at[:19].replace(" ", "T")  # ISO-compatible prefix
        if closed_at:
            ts_end = closed_at[:19].replace(" ", "T")
    except Exception:
        pass

    by_session: list[dict] = []
    by_time: list[dict] = []

    try:
        with router_log.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = _json.loads(line)
                except _json.JSONDecodeError:
                    continue

                # Exact session_id match
                if session_id and entry.get("session_id") == session_id:
                    by_session.append(entry)
                    continue  # no need to check time window

                # Time-window match (normalise "2026-04-15T20:57:48Z" → "2026-04-15T20:57:48")
                if ts_start:
                    raw_ts = (entry.get("ts") or "")[:19]
                    if raw_ts >= ts_start and (ts_end is None or raw_ts <= ts_end):
                        by_time.append(entry)
    except OSError:
        pass

    entries = by_session if by_session else by_time
    matched_by = "session_id" if by_session else ("time_window" if by_time else "none")

    model_counter: Counter = Counter()
    tier_counter: Counter = Counter()
    plan_mode_count = 0
    skills_counter: Counter = Counter()
    tokens_saved = 0
    compact_count = 0

    for entry in entries:
        verdict = entry.get("verdict") or {}
        model_counter[verdict.get("model", "unknown")] += 1
        # Support both old format (tier_used) and new format (tier)
        tier = verdict.get("tier_used") or verdict.get("tier") or 1
        tier_counter[str(tier)] += 1
        if verdict.get("plan_mode"):
            plan_mode_count += 1
        # Support both old format (preload_skills top-level) and new (skills.preloaded)
        skills_obj = entry.get("skills") or {}
        for sk in (skills_obj.get("preloaded") or entry.get("preload_skills") or []):
            if sk:
                skills_counter[sk] += 1
        tokens_saved += (entry.get("savings") or {}).get("tokens_estimated", 0) or 0
        if (entry.get("compact") or {}).get("suggested"):
            compact_count += 1

    total = sum(model_counter.values())
    models = [
        {
            "name": name,
            "count": count,
            "percentage": round(count / total * 100, 1) if total else 0.0,
        }
        for name, count in model_counter.most_common()
    ]
    return {
        "models": models,
        "plan_mode_count": plan_mode_count,
        "tier_counts": dict(tier_counter),
        "total_prompts": total,
        "top_skills": [{"name": n, "count": c} for n, c in skills_counter.most_common(5)],
        "matched_by": matched_by,
        "tokens_saved": tokens_saved,
        "compact_count": compact_count,
    }


@router.get("/tasks/{task_id}/stats/models")
def task_model_stats(task_id: int, request: Request) -> JSONResponse:
    """Return model routing stats for a task's window (from router.jsonl)."""
    store = request.app.state.store
    task = store.get_task(task_id)
    if not task:
        return JSONResponse({"error": "task not found"}, status_code=404)
    task_dict = dict(task)
    session_id = task_dict.get("session_id") or ""
    stats = _parse_router_log(
        session_id=session_id,
        created_at=task_dict.get("created_at"),
        closed_at=task_dict.get("closed_at"),
    )
    stats["session_id"] = session_id
    return JSONResponse(stats)


@router.get("/tasks/{task_id}/stats/skills")
def task_skills_stats(task_id: int, request: Request) -> JSONResponse:
    """Return top skills used during this task's window (from router.jsonl)."""
    store = request.app.state.store
    task = store.get_task(task_id)
    if not task:
        return JSONResponse({"error": "task not found"}, status_code=404)
    task_dict = dict(task)
    session_id = task_dict.get("session_id") or ""
    stats = _parse_router_log(
        session_id=session_id,
        created_at=task_dict.get("created_at"),
        closed_at=task_dict.get("closed_at"),
    )
    return JSONResponse({
        "top_skills": stats.get("top_skills", []),
        "total_prompts": stats.get("total_prompts", 0),
        "matched_by": stats.get("matched_by", "none"),
    })


@router.get("/tasks/{task_id}/stats/session")
def task_session_stats(task_id: int, request: Request) -> JSONResponse:
    """Return session-level stats: message count, duration, log line count."""
    store = request.app.state.store
    task = store.get_task(task_id)
    if not task:
        return JSONResponse({"error": "task not found"}, status_code=404)
    task_dict = dict(task)
    session_id = task_dict.get("session_id") or ""
    closed_at = task_dict.get("closed_at")
    created_at = task_dict.get("created_at")

    # Try exact session_id match in session_context (only works when the MCP
    # server UUID matches — legacy behaviour)
    message_count: int | None = None
    if session_id:
        try:
            ctx = store._conn.execute(
                "SELECT message_count FROM session_context WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if ctx:
                message_count = int(ctx["message_count"])
        except Exception:
            pass

    # Fallback: count router entries in the task time window.
    # The MCP server's internal session UUID differs from the Claude Code
    # conversation UUID, so the lookup above typically returns nothing.
    if message_count is None and created_at:
        try:
            router_stats = _parse_router_log(
                session_id=session_id,
                created_at=created_at,
                closed_at=closed_at,
            )
            if router_stats["total_prompts"] > 0:
                message_count = router_stats["total_prompts"]
        except Exception:
            pass

    duration_hours: float | None = None
    if closed_at and created_at:
        try:
            from datetime import datetime
            fmt = "%Y-%m-%d %H:%M:%S"
            dt_close = datetime.strptime(closed_at[:19], fmt)
            dt_create = datetime.strptime(created_at[:19], fmt)
            delta = (dt_close - dt_create).total_seconds() / 3600
            duration_hours = round(delta, 1) if delta >= 0.05 else None
        except Exception:
            pass

    log_count = 0
    try:
        import re as _re
        from ..services import log_tail as _lt
        needle = f"task={task_id}"
        _tag_re = _re.compile(r"\btask=(\d+)\b")
        for path in (_lt.HOOK_LOG, _lt.ACTIVITY_LOG):
            for ln in _lt.tail_file_sync(path, 5000):
                if needle in ln:
                    m = _tag_re.search(ln)
                    if m and m.group(1) == str(task_id):
                        log_count += 1
    except Exception:
        pass

    return JSONResponse({
        "session_id": session_id,
        "message_count": message_count,
        "duration_hours": duration_hours,
        "log_count": log_count,
        "updated_at": task_dict.get("updated_at"),
        "closed_at": closed_at,
        "auto_approve": task_dict.get("auto_approve"),
    })


@router.get("/tasks/refs")
def task_refs(request: Request, q: str = "") -> JSONResponse:
    """Reference picker data: tasks + skills matching q (for @-mention UI)."""
    store = request.app.state.store
    q = q.strip().lower()

    # Tasks — open first, then recent closed
    if q:
        rows = store.search_tasks_text(q, status="all")
        tasks = [{"id": r["id"], "title": r["title"], "status": r["status"]} for r in rows[:15]]
    else:
        open_rows = store.list_tasks(status="open")
        closed_rows = store.list_tasks(status="closed")
        tasks = [{"id": r["id"], "title": r["title"], "status": "open"} for r in open_rows[:10]]
        tasks += [{"id": r["id"], "title": r["title"], "status": "closed"} for r in closed_rows[:5]]

    # Skills — search by name/description substring
    try:
        if q:
            skill_rows = store._conn.execute(
                "SELECT id, name, description, plugin FROM skills "
                "WHERE lower(name) LIKE ? OR lower(description) LIKE ? ORDER BY name LIMIT 12",
                (f"%{q}%", f"%{q}%"),
            ).fetchall()
        else:
            skill_rows = store._conn.execute(
                "SELECT id, name, description, plugin FROM skills ORDER BY name LIMIT 12"
            ).fetchall()
        skills = [{"id": r["id"], "name": r["name"],
                   "description": (r["description"] or "")[:80],
                   "plugin": r["plugin"] or ""} for r in skill_rows]
    except Exception:
        skills = []

    # Articles — doc_id from vectors table (drafts, portfolio, memory docs)
    # Excludes session/habit/task namespaces; surfaces plugin-indexed content only.
    _article_ns_exclude = ("session:", "habits:", "tasks:", "skills")
    articles: list[dict] = []
    try:
        if q:
            art_rows = store._conn.execute(
                "SELECT doc_id, namespace, metadata FROM vectors "
                "WHERE lower(metadata) LIKE ? OR lower(doc_id) LIKE ? "
                "ORDER BY indexed_at DESC LIMIT 10",
                (f"%{q}%", f"%{q}%"),
            ).fetchall()
        else:
            art_rows = store._conn.execute(
                "SELECT doc_id, namespace, metadata FROM vectors "
                "WHERE namespace NOT LIKE 'session:%' AND namespace NOT LIKE 'habits:%' "
                "AND namespace NOT LIKE 'tasks:%' AND namespace != 'skills' "
                "ORDER BY indexed_at DESC LIMIT 10"
            ).fetchall()
        import json as _json
        for r in art_rows:
            ns = r["doc_id"] if r else ""
            skip = any(r["namespace"].startswith(p) for p in _article_ns_exclude)
            if skip:
                continue
            try:
                meta = _json.loads(r["metadata"] or "{}")
            except Exception:
                meta = {}
            label = meta.get("title") or meta.get("type") or ""
            doc_id = r["doc_id"]
            # Use filename stem as display name if path-like
            import os.path as _op
            display = label or _op.basename(doc_id).replace(".md", "").replace("-", " ")
            articles.append({
                "doc_id": doc_id,
                "namespace": r["namespace"],
                "display": display[:60],
            })
    except Exception:
        articles = []

    return JSONResponse({"tasks": tasks, "skills": skills, "articles": articles})


@router.post("/tasks/new")
async def create_task_web(request: Request) -> JSONResponse:
    """Create a task from the web UI (title required; summary/tags optional)."""
    form = await request.form()
    title = str(form.get("title", "")).strip()
    summary = str(form.get("summary", "")).strip()
    tags = str(form.get("tags", "")).strip()
    if not title:
        return JSONResponse({"error": "title required"}, status_code=400)
    store = request.app.state.store
    # Embed best-effort; fall back to empty vector
    vec: list[float] = _try_embed(f"{title} {summary}") or []
    task_id = store.save_task(title=title, summary=summary, tags=tags, vector=vec)
    return JSONResponse({"task_id": task_id, "title": title})
