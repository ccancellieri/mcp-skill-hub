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


@router.get("/tasks/{task_id}/stats/models")
def task_model_stats(task_id: int, request: Request) -> JSONResponse:
    """Return model usage and token statistics for a task's session."""
    store = request.app.state.store
    task = store.get_task(task_id)

    if not task:
        return JSONResponse({"error": "task not found"}, status_code=404)

    task_dict = dict(task)
    session_id = task_dict.get("session_id")

    # Default stats if no session
    if not session_id:
        return JSONResponse({
            "models": [],
            "total_tokens": 0,
            "session_id": None
        })

    # Query command_verdicts.db for model routing decisions in this session
    try:
        import sqlite3
        from pathlib import Path
        verdicts_db = Path.home() / ".claude" / "mcp-skill-hub" / "command_verdicts.db"

        if not verdicts_db.exists():
            return JSONResponse({
                "models": [],
                "total_tokens": 0,
                "session_id": session_id,
                "note": "verdicts database not found"
            })

        verdict_conn = sqlite3.connect(str(verdicts_db))
        verdict_conn.row_factory = sqlite3.Row

        # Aggregate model usage by tier within this session
        rows = verdict_conn.execute("""
            SELECT model, COUNT(*) as count, SUM(CAST(estimated_tokens AS INTEGER)) as tokens
            FROM command_verdicts
            WHERE session_id = ?
            GROUP BY model
            ORDER BY count DESC
        """, (session_id,)).fetchall()

        total_count = sum(dict(r)["count"] for r in rows)
        total_tokens = sum(dict(r)["tokens"] or 0 for r in rows)

        models = []
        for row in rows:
            d = dict(row)
            count = d.get("count", 0)
            percentage = (count / total_count * 100) if total_count > 0 else 0
            models.append({
                "name": d.get("model", "unknown"),
                "count": count,
                "percentage": round(percentage, 1),
                "tokens": d.get("tokens", 0) or 0
            })

        verdict_conn.close()

        return JSONResponse({
            "models": models,
            "total_tokens": total_tokens,
            "session_id": session_id
        })

    except Exception as e:
        return JSONResponse({
            "models": [],
            "total_tokens": 0,
            "session_id": session_id,
            "error": str(e)
        })
