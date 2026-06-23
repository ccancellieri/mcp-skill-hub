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

from ... import config as cfg
from ... import system_health as sh

router = APIRouter()

_COMPRESSION_EMPTY: dict = {
    "calls": 0, "hits": 0, "bytes_before": 0, "bytes_after": 0,
    "saved": 0, "avg_ratio": 1.0, "tokens_saved": 0,
    "by_strategy": {}, "by_site": {},
}

_LLM_STATS_EMPTY: dict = {
    "calls": 0, "errors": 0,
    "total_duration_ms": 0, "avg_latency_ms": 0.0,
    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
    "tokens_per_sec": 0.0,
    "by_op": {}, "by_model": {},
}


def _compression_context() -> dict:
    """Return compression stats + config flags for the health panel."""
    store = None
    try:
        from skill_hub.store import get_store
        store = get_store()
    except Exception:  # noqa: BLE001
        pass

    stats = _COMPRESSION_EMPTY
    llm_stats = _LLM_STATS_EMPTY
    if store is not None:
        try:
            stats = store.get_compression_stats()
        except Exception:  # noqa: BLE001
            pass
        try:
            llm_stats = store.get_llm_stats()
        except Exception:  # noqa: BLE001
            pass

    return {
        "compression_stats": stats,
        "compression_cfg": {
            "enabled": bool(cfg.get("compression_enabled")),
            "ml_enabled": bool(cfg.get("compression_ml_enabled")),
            "code_aware_enabled": bool(cfg.get("compression_code_aware_enabled")),
        },
        "llm_stats": llm_stats,
        "llm_metering_enabled": bool(cfg.get("llm_metering_enabled")),
    }


def _panel(request: Request, flash: str | None = None) -> HTMLResponse:
    snap = sh.health_snapshot()
    ctx: dict[str, Any] = {
        "snap": snap,
        "watcher": sh.watcher_status(),
        "flash": flash,
    }
    ctx.update(_compression_context())
    return request.app.state.templates.TemplateResponse(
        request,
        "_health_panel.html",
        ctx,
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


@router.post("/health/action/terminate", response_class=HTMLResponse)
async def action_terminate(request: Request) -> Any:
    """Gracefully terminate a single user process by pid (generic swap lever).

    Refuses system/root processes server-side, so this stays safe even though
    it can target any pid the swap-consumer ranking surfaces.
    """
    data = await _form_or_query(request)
    pid_raw = data.get("pid")
    if not pid_raw or not str(pid_raw).isdigit():
        return _panel(request, flash="⚠ no valid pid given")
    res = sh.terminate_process(int(pid_raw), hard=str(data.get("hard", "")).lower() in ("1", "true"))
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
