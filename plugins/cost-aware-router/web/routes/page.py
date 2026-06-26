"""Page routes: cost dashboard, settings."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from fastapi import APIRouter, Request

from cost_router import cost_tracker

router = APIRouter()
DB_PATH = cost_tracker.DEFAULT_DB_PATH


@router.get("/")
def index(request: Request):
    templates = request.app.state.templates
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    month_start = datetime.now(timezone.utc).strftime("%Y-%m-01")
    daily = cost_tracker.get_daily_cost(today)
    budget = cost_tracker.get_budget_status("global")

    monthly_cost = 0.0
    session_count = 0
    model_totals: list[dict] = []
    cheaper_alternatives: list[dict] = []

    if DB_PATH.exists():
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    """
                    SELECT SUM(cost_usd) as total
                    FROM plugin_cost_router_cost_log
                    WHERE date(created_at) >= ?
                    """,
                    (month_start,),
                ).fetchone()
                monthly_cost = row["total"] if row and row["total"] else 0.0

                row = conn.execute(
                    "SELECT COUNT(DISTINCT session_id) FROM plugin_cost_router_cost_log"
                ).fetchone()
                session_count = row[0] if row else 0

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

                for m in model_totals[:3]:
                    alt = cost_tracker.suggest_cheaper_alternative(m["model"], 0.5)
                    if alt:
                        cheaper_cost = cost_tracker.calculate_cost(
                            alt, m["in_tok"] or 0, m["out_tok"] or 0
                        )
                        cheaper_alternatives.append({
                            "current": m["model"],
                            "alternative": alt,
                            "current_cost": m["total"] or 0,
                            "alternative_cost": cheaper_cost,
                            "savings": (m["total"] or 0) - cheaper_cost,
                        })
        except sqlite3.Error:
            pass

    return templates.TemplateResponse(
        request,
        "page.html",
        {
            "active_tab": "cost-router",
            "daily": daily,
            "monthly_cost": monthly_cost,
            "session_count": session_count,
            "budget": budget,
            "model_totals": model_totals,
            "cheaper_alternatives": cheaper_alternatives,
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
