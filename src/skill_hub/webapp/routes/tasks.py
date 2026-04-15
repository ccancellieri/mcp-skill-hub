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


def _parse_router_log(session_id: str) -> dict:
    """Parse router.jsonl for model routing stats for a given session."""
    import json as _json
    from pathlib import Path
    from collections import Counter

    router_log = Path.home() / ".claude" / "mcp-skill-hub" / "router.jsonl"
    if not router_log.exists():
        return {"models": [], "plan_mode_count": 0, "tier_counts": {}, "total_prompts": 0, "top_skills": []}

    model_counter: Counter = Counter()
    tier_counter: Counter = Counter()
    plan_mode_count = 0
    skills_counter: Counter = Counter()

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
                if entry.get("session_id") != session_id:
                    continue
                verdict = entry.get("verdict") or {}
                model_counter[verdict.get("model", "unknown")] += 1
                tier_counter[str(verdict.get("tier_used", 0))] += 1
                if verdict.get("plan_mode"):
                    plan_mode_count += 1
                for sk in entry.get("preload_skills") or []:
                    if sk:
                        skills_counter[sk] += 1
    except OSError:
        pass

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
    }


@router.get("/tasks/{task_id}/stats/models")
def task_model_stats(task_id: int, request: Request) -> JSONResponse:
    """Return model routing stats for a task's session (from router.jsonl)."""
    store = request.app.state.store
    task = store.get_task(task_id)
    if not task:
        return JSONResponse({"error": "task not found"}, status_code=404)
    task_dict = dict(task)
    session_id = task_dict.get("session_id")
    if not session_id:
        return JSONResponse({"models": [], "total_prompts": 0, "session_id": None})
    stats = _parse_router_log(session_id)
    stats["session_id"] = session_id
    return JSONResponse(stats)


@router.get("/tasks/{task_id}/stats/session")
def task_session_stats(task_id: int, request: Request) -> JSONResponse:
    """Return session-level stats: message count, duration, log line count."""
    store = request.app.state.store
    task = store.get_task(task_id)
    if not task:
        return JSONResponse({"error": "task not found"}, status_code=404)
    task_dict = dict(task)
    session_id = task_dict.get("session_id")

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

    duration_hours: float | None = None
    closed_at = task_dict.get("closed_at")
    created_at = task_dict.get("created_at")
    if closed_at and created_at:
        try:
            from datetime import datetime
            fmt = "%Y-%m-%d %H:%M:%S"
            dt_close = datetime.strptime(closed_at[:19], fmt)
            dt_create = datetime.strptime(created_at[:19], fmt)
            duration_hours = round((dt_close - dt_create).total_seconds() / 3600, 1)
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

    return JSONResponse({"tasks": tasks, "skills": skills})


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
