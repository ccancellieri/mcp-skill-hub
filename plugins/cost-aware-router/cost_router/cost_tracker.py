"""Cost tracking core logic."""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path.home() / ".claude" / "mcp-skill-hub" / "skill_hub.db"


def get_pricing() -> dict[str, dict[str, float]]:
    """Load pricing from plugin config."""
    config_path = Path(__file__).parent.parent / "plugin.json"
    try:
        cfg = json.loads(config_path.read_text())
        return cfg.get("config", {}).get("pricing", {})
    except Exception:
        return {}


def resolve_model_name(raw: str) -> str:
    """Normalize model name to pricing key."""
    raw_lower = raw.lower()
    aliases = {
        "claude-3-5-sonnet": "sonnet",
        "claude-3-5-haiku": "haiku",
        "claude-3-opus": "opus",
        "claude-3-sonnet": "sonnet",
        "claude-3-haiku": "haiku",
        "claude-sonnet-4-20250514": "sonnet",
        "claude-opus-4-20250514": "opus",
        "claude-haiku-3-5-20241022": "haiku",
    }
    if raw_lower in aliases:
        return aliases[raw_lower]
    for pattern, key in [
        (r"claude.*opus", "opus"),
        (r"claude.*sonnet", "sonnet"),
        (r"claude.*haiku", "haiku"),
        (r"deepseek.*r1", "deepseek-r1"),
    ]:
        if re.search(pattern, raw_lower):
            return key
    return raw_lower


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for given model and token counts."""
    pricing = get_pricing()
    resolved = resolve_model_name(model)
    rates = pricing.get(resolved, {"input": 0.0, "output": 0.0})
    input_cost = (input_tokens * rates.get("input", 0.0)) / 1_000_000
    output_cost = (output_tokens * rates.get("output", 0.0)) / 1_000_000
    return input_cost + output_cost


def log_cost(
    session_id: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    project: str | None = None,
    tool_name: str | None = None,
    tier: str | None = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    """Insert a cost log entry. Returns row id."""
    if not db_path.exists():
        return 0
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO plugin_cost_router_cost_log
                    (session_id, project, model, input_tokens, output_tokens,
                     cost_usd, tool_name, tier)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, project, model, input_tokens, output_tokens,
                 cost_usd, tool_name, tier),
            )
            conn.commit()
            return cur.lastrowid or 0
    except sqlite3.Error:
        return 0


def get_session_cost(session_id: str, db_path: Path = DEFAULT_DB_PATH) -> dict[str, Any]:
    """Get total cost for a session."""
    if not db_path.exists():
        return {"total_usd": 0.0, "by_model": {}}
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT model, SUM(cost_usd) as total, SUM(input_tokens) as in_tok,
                       SUM(output_tokens) as out_tok
                FROM plugin_cost_router_cost_log
                WHERE session_id = ?
                GROUP BY model
                """,
                (session_id,),
            ).fetchall()
        total = sum(r["total"] or 0 for r in rows)
        by_model = {r["model"]: {"cost": r["total"], "input": r["in_tok"], "output": r["out_tok"]} for r in rows}
        return {"total_usd": total, "by_model": by_model, "calls": len(rows)}
    except sqlite3.Error:
        return {"total_usd": 0.0, "by_model": {}}


def get_daily_cost(date: str | None = None, db_path: Path = DEFAULT_DB_PATH) -> dict[str, Any]:
    """Get total cost for a date (YYYY-MM-DD). Defaults to today."""
    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if not db_path.exists():
        return {"date": date, "total_usd": 0.0, "by_model": {}}
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT model, SUM(cost_usd) as total, SUM(input_tokens) as in_tok,
                       SUM(output_tokens) as out_tok, COUNT(*) as calls
                FROM plugin_cost_router_cost_log
                WHERE date(created_at) = ?
                GROUP BY model
                """,
                (date,),
            ).fetchall()
        total = sum(r["total"] or 0 for r in rows)
        by_model = {
            r["model"]: {
                "cost": r["total"],
                "input": r["in_tok"],
                "output": r["out_tok"],
                "calls": r["calls"],
            }
            for r in rows
        }
        return {"date": date, "total_usd": total, "by_model": by_model}
    except sqlite3.Error:
        return {"date": date, "total_usd": 0.0, "by_model": {}}


def get_budget_status(scope: str, db_path: Path = DEFAULT_DB_PATH) -> dict[str, Any]:
    """Get budget status for a scope (session/project/daily/global)."""
    if not db_path.exists():
        return {"budget_usd": 10.0, "spent_usd": 0.0, "remaining_usd": 10.0, "pct_used": 0.0}
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT budget_usd, spent_usd FROM plugin_cost_router_budget_limits WHERE scope = ?",
                (scope,),
            ).fetchone()
        if row:
            budget = row["budget_usd"]
            spent = row["spent_usd"]
            return {
                "budget_usd": budget,
                "spent_usd": spent,
                "remaining_usd": max(0, budget - spent),
                "pct_used": (spent / budget * 100) if budget > 0 else 0,
            }
        config_path = Path(__file__).parent.parent / "plugin.json"
        default = 10.0
        try:
            cfg = json.loads(config_path.read_text())
            default = cfg.get("config", {}).get("default_budget_usd", 10.0)
        except Exception:
            pass
        return {"budget_usd": default, "spent_usd": 0.0, "remaining_usd": default, "pct_used": 0.0}
    except sqlite3.Error:
        return {"budget_usd": 10.0, "spent_usd": 0.0, "remaining_usd": 10.0, "pct_used": 0.0}


def set_budget(scope: str, scope_type: str, budget_usd: float, db_path: Path = DEFAULT_DB_PATH) -> bool:
    """Set or update a budget limit."""
    if not db_path.exists():
        return False
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO plugin_cost_router_budget_limits (scope, scope_type, budget_usd)
                VALUES (?, ?, ?)
                ON CONFLICT(scope) DO UPDATE SET budget_usd = excluded.budget_usd,
                                                  updated_at = datetime('now')
                """,
                (scope, scope_type, budget_usd),
            )
            conn.commit()
            return True
    except sqlite3.Error:
        return False


def suggest_cheaper_alternative(current_model: str, complexity: float = 0.5) -> str | None:
    """Suggest a cheaper model if current is expensive for the task."""
    hierarchy = ["haiku", "sonnet", "opus"]
    resolved = resolve_model_name(current_model)
    if resolved not in hierarchy:
        return None
    idx = hierarchy.index(resolved)
    if idx == 0:
        return None
    if complexity < 0.3 and idx > 0:
        return hierarchy[idx - 1]
    if complexity < 0.6 and idx > 1:
        return hierarchy[idx - 1]
    return None
