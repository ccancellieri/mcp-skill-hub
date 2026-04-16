"""Cron Jobs panel — CRUD + run-now + HTMX fragments."""
from __future__ import annotations

import json
import threading
from typing import Any

from croniter import croniter as _croniter
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
    if not _croniter.is_valid(schedule):
        return JSONResponse(
            {"error": f"invalid cron expression: {schedule!r}"}, status_code=400
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
    # Disallow name changes in PATCH — name is the natural key
    if "name" in body and body["name"] != job["name"]:
        return JSONResponse({"error": "cannot change job name"}, status_code=400)
    # Apply updates
    enabled = body.get("enabled", job.get("enabled", 1))
    if "enabled" in body:
        store.toggle_cron_job(job_id, bool(enabled))
    # For schedule / command / description: update by id, not by name
    if any(k in body for k in ("schedule", "command", "description", "params")):
        if job.get("is_builtin"):
            return JSONResponse(
                {"error": "cannot modify builtin job fields (use toggle to enable/disable)"},
                status_code=403,
            )
        new_schedule = body.get("schedule", job.get("schedule", ""))
        if new_schedule and not _croniter.is_valid(new_schedule):
            return JSONResponse(
                {"error": f"invalid cron expression: {new_schedule!r}"}, status_code=400
            )
        store._conn.execute(
            "UPDATE cron_jobs SET schedule=?, command=?, enabled=?, description=?,"
            " updated_at=datetime('now') WHERE id=?",
            (
                new_schedule,
                body.get("command", job.get("command", "")),
                int(bool(body.get("enabled", job.get("enabled", 1)))),
                body.get("description", job.get("description", "") or ""),
                job_id,
            ),
        )
        store._conn.commit()
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
    """Force a job to run immediately (background thread) and also reset last_run_at."""
    store = request.app.state.store
    row = store.get_cron_job(job_id)
    if row is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    # Reset last_run_at to epoch so croniter considers it overdue on next tick too.
    store.reset_cron_job_for_run(job_id)
    # Also fire immediately in a background thread — don't wait for the next tick.
    job_dict = _row_to_dict(row)

    def _run() -> None:
        try:
            _cron.get_scheduler()._run_job(job_dict)
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()
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
