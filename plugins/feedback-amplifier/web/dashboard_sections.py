"""Dashboard KPI card for the feedback-amplifier plugin.

Returns one section showing total feedback events, average skill score,
and counts of boosted/decayed skills.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

_PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

HUB_ROOT = _PLUGIN_ROOT.parent.parent
DB_PATH = HUB_ROOT / "skill_hub.db"


def _get_plugin_config() -> dict:
    try:
        if not DB_PATH.exists():
            return {}
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT value FROM config WHERE key = 'plugins'"
            ).fetchone()
            if row:
                plugins = json.loads(row["value"])
                for p in plugins:
                    if p.get("path", "").endswith("feedback-amplifier"):
                        return p.get("config", {})
    except Exception:
        pass
    return {}


def get_sections() -> list[dict]:
    total_feedback = 0
    avg_score = 0.0
    skills_boosted = 0
    skills_decayed = 0

    if DB_PATH.exists():
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row

                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM plugin_fbamp_feedback_context"
                ).fetchone()
                total_feedback = row["cnt"] if row else 0

                rows = conn.execute("""
                    SELECT ema_score FROM plugin_fbamp_skill_scores
                """).fetchall()

                if rows:
                    total = sum(r["ema_score"] for r in rows)
                    avg_score = round(total / len(rows), 2)
                    for r in rows:
                        if r["ema_score"] > 1.1:
                            skills_boosted += 1
                        elif r["ema_score"] < 0.9:
                            skills_decayed += 1

        except sqlite3.Error:
            pass

    return [
        {
            "id": "feedback-amplifier",
            "title": "Feedback Amplifier",
            "order": 70,
            "template": "dashboard_kpi.html",
            "context": {
                "total_feedback": total_feedback,
                "avg_score": avg_score,
                "skills_boosted": skills_boosted,
                "skills_decayed": skills_decayed,
            },
        }
    ]
