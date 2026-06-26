"""Page routes: cost dashboard, settings."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Request

from cost_router import cost_tracker

router = APIRouter()
DB_PATH = cost_tracker.DEFAULT_DB_PATH


@router.get("/")
def index(request: Request):
    templates = request.app.state.templates
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    daily = cost_tracker.get_daily_cost(today)
    session_id = request.query_params.get("session", "")
    session_cost = cost_tracker.get_session_cost(session_id) if session_id else None
    budget = cost_tracker.get_budget_status("global")

    recent_sessions: list[dict] = []
    model_totals: list[dict] = []
    if DB_PATH.exists():
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT session_id, SUM(cost_usd) as total, COUNT(*) as calls,
                           MAX(created_at) as last_call
                    FROM plugin_cost_router_cost_log
                    GROUP BY session_id
                    ORDER BY last_call DESC
                    LIMIT 10
                    """
                ).fetchall()
                recent_sessions = [dict(r) for r in rows]
                mrows = conn.execute(
                    """
                    SELECT model, SUM(cost_usd) as total, SUM(input_tokens) as in_tok,
                           SUM(output_tokens) as out_tok, COUNT(*) as calls
                    FROM plugin_cost_router_cost_log
                    GROUP BY model
                    ORDER BY total DESC
                    """
                ).fetchall()
                model_totals = [dict(r) for r in mrows]
        except sqlite3.Error:
            pass

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "active_tab": "cost-router",
            "daily": daily,
            "session_cost": session_cost,
            "budget": budget,
            "recent_sessions": recent_sessions,
            "model_totals": model_totals,
            "session_id": session_id,
        },
    )


@router.get("/settings")
def settings_page(request: Request):
    templates = request.app.state.templates
    budgets: list[dict] = []
    if DB_PATH.exists():
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT scope, scope_type, budget_usd, spent_usd FROM plugin_cost_router_budget_limits"
                ).fetchall()
                budgets = [dict(r) for r in rows]
        except sqlite3.Error:
            pass
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "active_tab": "cost-router",
            "budgets": budgets,
        },
    )
