"""Dashboard KPI card for the cost-aware-router plugin."""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

_PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

from cost_router import cost_tracker

DB_PATH = cost_tracker.DEFAULT_DB_PATH


def get_sections() -> list[dict]:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    daily = cost_tracker.get_daily_cost(today)
    budget = cost_tracker.get_budget_status("global")

    session_count = 0
    if DB_PATH.exists():
        try:
            with sqlite3.connect(DB_PATH) as conn:
                row = conn.execute(
                    "SELECT COUNT(DISTINCT session_id) FROM plugin_cost_router_cost_log"
                ).fetchone()
                session_count = row[0] if row else 0
        except sqlite3.Error:
            pass

    return [
        {
            "id": "cost-router",
            "title": "Cost Router",
            "order": 70,
            "template": "dashboard_kpi.html",
            "context": {
                "daily_cost": daily.get("total_usd", 0.0),
                "budget_pct": budget.get("pct_used", 0.0),
                "budget_remaining": budget.get("remaining_usd", budget.get("budget_usd", 10.0)),
                "session_count": session_count,
            },
        }
    ]
