"""System Health — resource panel with one-click remediation.

Surfaces the same diagnosis a human would do by hand (swap, RAM, CPU load,
runaway Claude Code daemons, long-running Docker stacks) and turns each finding
into a button. Read-only probes live in :mod:`skill_hub.system_health`; this
module is the thin HTTP layer (page + polled fragment + POST actions).
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ... import system_health as sh

router = APIRouter()


def _panel(request: Request, flash: str | None = None) -> HTMLResponse:
    snap = sh.health_snapshot()
    return request.app.state.templates.TemplateResponse(
        request,
        "_health_panel.html",
        {
            "snap": snap,
            "watcher": sh.watcher_status(),
            "flash": flash,
        },
    )


@router.get("/health", response_class=HTMLResponse)
def health_page(request: Request) -> Any:
    return request.app.state.templates.TemplateResponse(
        request, "health.html", {}
    )


@router.get("/health/panel", response_class=HTMLResponse)
def health_panel(request: Request) -> Any:
    return _panel(request)


@router.get("/health/json", response_class=JSONResponse)
def health_json() -> Any:
    return sh.health_snapshot().to_dict()


@router.post("/health/action/kill-daemons", response_class=HTMLResponse)
async def action_kill_daemons(request: Request) -> Any:
    """Kill stale Claude processes. With ``?pids=`` (comma list), kill those."""
    pids_raw = (await _form_or_query(request)).get("pids")
    if pids_raw:
        pids = [int(p) for p in str(pids_raw).split(",") if p.strip().isdigit()]
        res = sh.kill_claude(pids)
    else:
        res = sh.kill_stale_claude()
    flash = _fmt_kill(res)
    return _panel(request, flash=flash)


@router.post("/health/action/stop-docker", response_class=HTMLResponse)
async def action_stop_docker(request: Request) -> Any:
    """Stop containers. With ``?names=`` (comma list), stop those; else all."""
    names_raw = (await _form_or_query(request)).get("names")
    names = (
        [n for n in str(names_raw).split(",") if n.strip()]
        if names_raw else None
    )
    res = sh.stop_docker(names)
    return _panel(request, flash=_fmt_docker(res))


@router.post("/health/action/stop-nonessential", response_class=HTMLResponse)
async def action_stop_nonessential(request: Request) -> Any:
    """Stop every container except the catalog/DB/Elasticsearch working set."""
    res = sh.stop_nonessential_docker()
    return _panel(request, flash=_fmt_docker(res))


def _fmt_docker(res: dict) -> str:
    if res.get("error"):
        return f"Docker stop failed: {res['error']}"
    stopped = res.get("stopped") or []
    kept = res.get("kept") or []
    if not stopped:
        return res.get("note") or "Nothing to stop"
    msg = f"Stopped {len(stopped)}: {', '.join(stopped)}"
    if kept:
        msg += f" — kept {', '.join(kept)}"
    return msg


@router.post("/health/action/purge-memory", response_class=HTMLResponse)
async def action_purge_memory(request: Request) -> Any:
    res = sh.purge_memory()
    flash = ("✓ " if res.get("ok") else "⚠ ") + str(res.get("note") or "")
    return _panel(request, flash=flash)


def _fmt_kill(res: dict) -> str:
    if res.get("note") and not res.get("killed"):
        return res["note"]
    parts = []
    if res.get("killed"):
        parts.append(f"killed {res['killed']}")
    if res.get("failed"):
        parts.append(f"failed {res['failed']}")
    if res.get("skipped"):
        parts.append(f"skipped {len(res['skipped'])}")
    return "; ".join(parts) or "nothing to do"


async def _form_or_query(request: Request) -> dict:
    """Accept params from either form body or query string (HTMX sends form)."""
    data: dict[str, Any] = dict(request.query_params)
    ctype = request.headers.get("content-type", "")
    if "application/x-www-form-urlencoded" in ctype or "multipart/form-data" in ctype:
        form = await request.form()
        data.update({k: v for k, v in form.items()})
    return data
