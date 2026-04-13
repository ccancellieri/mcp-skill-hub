"""Dashboard route — KPIs, metrics, and home page."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ... import dashboard as _dashboard
from ... import dashboard_api  # noqa: F401 (plan reference; future use)

router = APIRouter()


def _collect_metrics(store: Any) -> dict[str, Any]:
    db = _dashboard._db_metrics(store)
    logm = _dashboard._parse_log()
    vcache = _dashboard._verdict_metrics()
    tokens_saved = int(db.get("tokens_saved") or 0)
    llm_seconds = sum(logm["llm_ms"]) / 1000.0
    llm_cost_eq = int(llm_seconds * _dashboard.TOKENS_PER_LLM_SECOND)
    net = tokens_saved - llm_cost_eq
    tasks_open = db["tasks"].get("open", 0) if isinstance(db.get("tasks"), dict) else 0
    tasks_closed = db["tasks"].get("closed", 0) if isinstance(db.get("tasks"), dict) else 0
    helpful = db.get("feedback_helpful", 0)
    unhelpful = db.get("feedback_unhelpful", 0)
    total_fb = helpful + unhelpful
    helpful_pct = (helpful / total_fb * 100.0) if total_fb else 0.0
    approve = logm["auto_approve"].get("allow", 0)
    deny = logm["auto_approve"].get("deny", 0)
    pass_through = logm["auto_approve"].get("pass", 0)
    return {
        "tokens_saved": tokens_saved,
        "llm_cost_eq": llm_cost_eq,
        "net": net,
        "tasks_open": tasks_open,
        "tasks_closed": tasks_closed,
        "skills": db.get("skills", 0),
        "teachings": db.get("teachings", 0),
        "helpful": helpful,
        "unhelpful": unhelpful,
        "helpful_pct": round(helpful_pct, 1),
        "approve": approve,
        "deny": deny,
        "pass_through": pass_through,
        "auto_proceed_fires": logm.get("auto_proceed_fires", 0),
        "resume_consumed": logm.get("resume_consumed", 0),
        "intercept_errors": logm.get("intercept_errors", 0),
        "llm_samples": len(logm["llm_ms"]),
        "verdict_total": vcache.get("total", 0),
        "verdict_hits": vcache.get("hits_total", 0),
        "log_missing": logm.get("log_missing", False),
    }


@router.get("/api/metrics")
def api_metrics(request: Request) -> JSONResponse:
    store = request.app.state.store
    return JSONResponse(_collect_metrics(store))


@router.get("/", response_class=HTMLResponse)
def index(request: Request) -> Any:
    store = request.app.state.store
    metrics = _collect_metrics(store)
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {"m": metrics, "active_tab": "dashboard"},
    )
