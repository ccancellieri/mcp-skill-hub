#!/usr/bin/env python3
"""A3 hook: on_tool_call — track model invocations and token usage.

Receives JSON on stdin with:
  - event: "on_tool_call"
  - tool_name: the MCP tool being called
  - session_id: current session ID
  - plugin_id: optional plugin identifier

For LLM-related tools, parses output for usage stats and logs costs.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path.home() / ".claude" / "mcp-skill-hub" / "skill_hub.db"

LLM_TOOLS = {
    "route_to_model",
    "improve_prompt",
    "search_web",
    "wiki_file_answer",
    "wiki_ingest",
    "optimize_memory",
    "optimize_plugin_memory",
}


def _load_pricing() -> dict[str, dict[str, float]]:
    cfg_path = PLUGIN_ROOT / "plugin.json"
    try:
        cfg = json.loads(cfg_path.read_text())
        return cfg.get("config", {}).get("pricing", {})
    except Exception:
        return {}


def _resolve_model(raw: str) -> str:
    raw_lower = raw.lower()
    aliases = {
        "claude-3-5-sonnet": "sonnet",
        "claude-3-5-haiku": "haiku",
        "claude-3-opus": "opus",
        "claude-sonnet-4-20250514": "sonnet",
        "claude-opus-4-20250514": "opus",
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


def _calc_cost(model: str, in_tok: int, out_tok: int) -> float:
    pricing = _load_pricing()
    rates = pricing.get(model, {"input": 0.0, "output": 0.0})
    return (in_tok * rates.get("input", 0.0) + out_tok * rates.get("output", 0.0)) / 1_000_000


def _log_cost(
    session_id: str,
    model: str,
    in_tok: int,
    out_tok: int,
    cost: float,
    tool_name: str | None = None,
) -> None:
    if not DB_PATH.exists():
        return
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO plugin_cost_router_cost_log
                    (session_id, model, input_tokens, output_tokens, cost_usd, tool_name)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, model, in_tok, out_tok, cost, tool_name),
            )
            conn.commit()
    except sqlite3.Error:
        pass


def _check_budget_alert(session_id: str) -> str | None:
    if not DB_PATH.exists():
        return None
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT b.budget_usd, COALESCE(SUM(l.cost_usd), 0) as spent
                FROM plugin_cost_router_budget_limits b
                LEFT JOIN plugin_cost_router_cost_log l ON l.session_id = b.scope
                WHERE b.scope = ? AND b.scope_type = 'session'
                GROUP BY b.scope
                """,
                (session_id,),
            ).fetchone()
            if not row:
                return None
            pct = (row["spent"] / row["budget_usd"] * 100) if row["budget_usd"] > 0 else 0
            if pct >= 80:
                return f"Budget alert: {pct:.0f}% of session budget used (${row['spent']:.2f} / ${row['budget_usd']:.2f})"
            return None
    except sqlite3.Error:
        return None


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(1)

    tool_name = payload.get("tool_name", "")
    session_id = payload.get("session_id", "")

    if tool_name not in LLM_TOOLS:
        sys.exit(0)

    alert = _check_budget_alert(session_id)
    if alert:
        print(json.dumps({"alert": alert}))

    sys.exit(0)


if __name__ == "__main__":
    main()
