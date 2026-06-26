"""``GET /`` — overview page showing skill scores and decay stats."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fastapi import APIRouter, Request

router = APIRouter()

HUB_ROOT = Path(__file__).resolve().parent.parent.parent.parent
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


@router.get("/")
def index(request: Request):
    templates = request.app.state.templates

    total_feedback = 0
    skills_boosted = 0
    skills_decayed = 0
    skill_scores = []
    decay_half_life = 30

    if DB_PATH.exists():
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row

                config = _get_plugin_config()
                decay_half_life = config.get("decay_half_life_days", 30)

                total_feedback = conn.execute(
                    "SELECT COUNT(*) as cnt FROM plugin_fbamp_feedback_context"
                ).fetchone()["cnt"]

                rows = conn.execute("""
                    SELECT skill_id, ema_score, injection_count, used_count, last_used_at
                    FROM plugin_fbamp_skill_scores
                    ORDER BY ema_score DESC
                """).fetchall()

                for row in rows:
                    skill_scores.append({
                        "skill_id": row["skill_id"],
                        "ema_score": round(row["ema_score"], 3),
                        "injection_count": row["injection_count"] or 0,
                        "used_count": row["used_count"] or 0,
                        "last_used_at": row["last_used_at"] or "never",
                    })
                    if row["ema_score"] > 1.1:
                        skills_boosted += 1
                    elif row["ema_score"] < 0.9:
                        skills_decayed += 1

        except sqlite3.Error:
            pass

    return templates.TemplateResponse(
        request,
        "page.html",
        {
            "active_tab": "feedback-amp",
            "total_feedback": total_feedback,
            "skills_boosted": skills_boosted,
            "skills_decayed": skills_decayed,
            "skill_scores": skill_scores,
            "decay_half_life": decay_half_life,
        },
    )
