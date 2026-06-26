"""REST endpoints: cost queries, budget management."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from cost_router import cost_tracker

router = APIRouter()
DB_PATH = cost_tracker.DEFAULT_DB_PATH


@router.get("/api/cost/session/{session_id}")
def session_cost(session_id: str):
    return JSONResponse(cost_tracker.get_session_cost(session_id))


@router.get("/api/cost/daily")
def daily_cost(date: Optional[str] = Query(None, regex=r"^\d{4}-\d{2}-\d{2}$")):
    return JSONResponse(cost_tracker.get_daily_cost(date))


@router.get("/api/cost/range")
def cost_range(
    start: str = Query(..., regex=r"^\d{4}-\d{2}-\d{2}$"),
    end: str = Query(..., regex=r"^\d{4}-\d{2}-\d{2}$"),
):
    if not DB_PATH.exists():
        return JSONResponse({"days": [], "total_usd": 0.0})
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT date(created_at) as day, SUM(cost_usd) as total,
                       SUM(input_tokens) as in_tok, SUM(output_tokens) as out_tok
                FROM plugin_cost_router_cost_log
                WHERE date(created_at) BETWEEN ? AND ?
                GROUP BY day
                ORDER BY day
                """,
                (start, end),
            ).fetchall()
        days = [dict(r) for r in rows]
        total = sum(r["total"] or 0 for r in rows)
        return JSONResponse({"days": days, "total_usd": total})
    except sqlite3.Error:
        return JSONResponse({"days": [], "total_usd": 0.0})


@router.get("/api/cost/by-model")
def cost_by_model(
    start: Optional[str] = Query(None, regex=r"^\d{4}-\d{2}-\d{2}$"),
    end: Optional[str] = Query(None, regex=r"^\d{4}-\d{2}-\d{2}$"),
):
    if not DB_PATH.exists():
        return JSONResponse({"models": []})
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            sql = """
                SELECT model, SUM(cost_usd) as total,
                       SUM(input_tokens) as in_tok, SUM(output_tokens) as out_tok,
                       COUNT(*) as calls
                FROM plugin_cost_router_cost_log
            """
            params = []
            if start and end:
                sql += " WHERE date(created_at) BETWEEN ? AND ?"
                params = [start, end]
            sql += " GROUP BY model ORDER BY total DESC"
            rows = conn.execute(sql, params).fetchall()
        return JSONResponse({"models": [dict(r) for r in rows]})
    except sqlite3.Error:
        return JSONResponse({"models": []})


@router.get("/api/budget/{scope}")
def get_budget(scope: str):
    return JSONResponse(cost_tracker.get_budget_status(scope))


@router.post("/api/budget/{scope}")
def set_budget(scope: str, budget_usd: float = Query(..., gt=0)):
    scope_type = "global" if scope == "global" else "session"
    ok = cost_tracker.set_budget(scope, scope_type, budget_usd)
    if not ok:
        raise HTTPException(500, "Failed to set budget")
    return JSONResponse({"ok": True, "scope": scope, "budget_usd": budget_usd})


@router.get("/api/estimate")
def estimate_cost(
    model: str = Query(...),
    input_tokens: int = Query(..., ge=0),
    output_tokens: int = Query(..., ge=0),
):
    resolved = cost_tracker.resolve_model_name(model)
    cost = cost_tracker.calculate_cost(model, input_tokens, output_tokens)
    cheaper = cost_tracker.suggest_cheaper_alternative(model, 0.5)
    response = {
        "model": resolved,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
    }
    if cheaper:
        cheaper_cost = cost_tracker.calculate_cost(cheaper, input_tokens, output_tokens)
        response["cheaper_alternative"] = {
            "model": cheaper,
            "cost_usd": cheaper_cost,
            "savings_usd": cost - cheaper_cost,
        }
    return JSONResponse(response)


@router.get("/api/pricing")
def get_pricing():
    return JSONResponse(cost_tracker.get_pricing())
