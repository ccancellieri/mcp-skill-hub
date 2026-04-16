"""Cron Jobs panel — CRUD + run-now + HTMX fragments."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ... import cron as _cron

router = APIRouter()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row) -> dict:
    if row is None:
        return {}
    d = dict(row)
    try:
        d["params"] = json.loads(d.get("params") or "{}")
    except Exception:
        d["params"] = {}
    return d


def _enrich(job: dict) -> dict:
    """Add computed display fields to a job dict."""
    job["human_schedule"] = _cron.human_schedule(job.get("schedule") or "")
    last = job.get("last_run_at")
    last_dt = None
    if last:
        try:
            from datetime import datetime, timezone
            last_dt = datetime.fromisoformat(last)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    nxt = _cron.next_run_from(job.get("schedule") or "", last_dt)
    job["next_run_at"] = nxt.isoformat() if nxt else None
    return job


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------


@router.get("/cron", response_class=HTMLResponse)
def cron_page(request: Request) -> Any:
    store = request.app.state.store
    jobs = [_enrich(_row_to_dict(r)) for r in store.list_cron_jobs()]
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "cron.html",
        {
            "active_tab": "cron",
            "jobs": jobs,
            "total": len(jobs),
        },
    )


# ---------------------------------------------------------------------------
# HTMX fragment — jobs table body
# ---------------------------------------------------------------------------


@router.get("/cron/jobs-table", response_class=HTMLResponse)
def cron_jobs_table(request: Request) -> Any:
    store = request.app.state.store
    jobs = [_enrich(_row_to_dict(r)) for r in store.list_cron_jobs()]
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "_cron_rows.html",
        {"jobs": jobs},
    )


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------


@router.get("/api/cron")
def api_cron_list(request: Request) -> JSONResponse:
    store = request.app.state.store
    jobs = [_enrich(_row_to_dict(r)) for r in store.list_cron_jobs()]
    return JSONResponse(jobs)


@router.post("/api/cron")
async def api_cron_create(request: Request) -> JSONResponse:
    body = await request.json()
    name = (body.get("name") or "").strip()
    schedule = (body.get("schedule") or "").strip()
    command = (body.get("command") or "").strip()
    if not name or not schedule or not command:
        return JSONResponse(
            {"error": "name, schedule, and command are required"}, status_code=400
        )
    store = request.app.state.store
    try:
        job_id = store.upsert_cron_job(
            name=name,
            description=body.get("description", ""),
            schedule=schedule,
            command=command,
            params=body.get("params", {}),
            enabled=bool(body.get("enabled", True)),
            is_builtin=False,
            is_dangerous=bool(body.get("is_dangerous", False)),
        )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    row = store.get_cron_job(job_id)
    job = _enrich(_row_to_dict(row)) if row else {}
    return JSONResponse(job, status_code=201)


@router.patch("/api/cron/{job_id}")
async def api_cron_update(job_id: int, request: Request) -> JSONResponse:
    body = await request.json()
    store = request.app.state.store
    row = store.get_cron_job(job_id)
    if row is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    job = _row_to_dict(row)
    # Apply updates
    enabled = body.get("enabled", job.get("enabled", 1))
    if "enabled" in body:
        store.toggle_cron_job(job_id, bool(enabled))
    # For schedule / name / command: rebuild via upsert using current values
    if any(k in body for k in ("schedule", "name", "command", "description", "params")):
        store.upsert_cron_job(
            name=body.get("name", job.get("name", "")),
            description=body.get("description", job.get("description", "") or ""),
            schedule=body.get("schedule", job.get("schedule", "")),
            command=body.get("command", job.get("command", "")),
            params=body.get("params", job.get("params", {})),
            enabled=bool(body.get("enabled", job.get("enabled", 1))),
            is_builtin=bool(job.get("is_builtin", 0)),
            is_dangerous=bool(job.get("is_dangerous", 0)),
        )
    row = store.get_cron_job(job_id)
    return JSONResponse(_enrich(_row_to_dict(row)) if row else {})


@router.delete("/api/cron/{job_id}")
def api_cron_delete(job_id: int, request: Request) -> JSONResponse:
    store = request.app.state.store
    deleted = store.delete_cron_job(job_id)
    if not deleted:
        return JSONResponse(
            {"error": "cannot delete builtin job or job not found"}, status_code=403
        )
    return JSONResponse({"ok": True})


@router.post("/api/cron/{job_id}/run-now")
def api_cron_run_now(job_id: int, request: Request) -> JSONResponse:
    """Force a job to run on the next scheduler tick by resetting last_run_at."""
    store = request.app.state.store
    row = store.get_cron_job(job_id)
    if row is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    # Reset last_run_at so the scheduler will fire it on the next tick.
    store.update_cron_job_status(job_id, "pending")
    # Also reset last_run_at to epoch so croniter considers it overdue
    store._conn.execute(
        "UPDATE cron_jobs SET last_run_at='2000-01-01T00:00:00' WHERE id=?",
        (job_id,),
    )
    store._conn.commit()
    row = store.get_cron_job(job_id)
    return JSONResponse(_enrich(_row_to_dict(row)) if row else {})


@router.patch("/api/cron/{job_id}/toggle", response_class=HTMLResponse)
def api_cron_toggle(job_id: int, request: Request) -> HTMLResponse:
    store = request.app.state.store
    row = store.get_cron_job(job_id)
    if row is None:
        return HTMLResponse("<span class='tag deny'>not found</span>", 404)
    new_enabled = not bool(row["enabled"])
    store.toggle_cron_job(job_id, new_enabled)
    label = "active" if new_enabled else "paused"
    css = "allow" if new_enabled else "closed"
    return HTMLResponse(
        f"<span class='tag {css}' style='cursor:pointer'"
        f" hx-patch='/api/cron/{job_id}/toggle' hx-swap='outerHTML'>{label}</span>"
    )
