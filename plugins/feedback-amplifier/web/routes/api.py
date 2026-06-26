"""REST endpoints: scores, stats, recompute trigger."""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()

HUB_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
DB_PATH = HUB_ROOT / "skill_hub.db"
SCRIPT_PATH = (
    HUB_ROOT / "plugins" / "feedback-amplifier" / "scripts" / "compute_scores.py"
)


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


@router.get("/api/scores")
def list_scores():
    if not DB_PATH.exists():
        return JSONResponse({"scores": []})
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT skill_id, ema_score, injection_count, used_count, last_used_at
                FROM plugin_fbamp_skill_scores
                ORDER BY ema_score DESC
            """).fetchall()
            return JSONResponse({
                "scores": [
                    {
                        "skill_id": r["skill_id"],
                        "ema_score": round(r["ema_score"], 4),
                        "injection_count": r["injection_count"] or 0,
                        "used_count": r["used_count"] or 0,
                        "last_used_at": r["last_used_at"] or None,
                    }
                    for r in rows
                ]
            })
    except sqlite3.Error as e:
        raise HTTPException(500, f"Database error: {e}") from e


@router.get("/api/stats")
def get_stats():
    total_feedback = 0
    skills_boosted = 0
    skills_decayed = 0
    avg_score = 0.0
    decay_half_life = 30

    if DB_PATH.exists():
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row

                config = _get_plugin_config()
                decay_half_life = config.get("decay_half_life_days", 30)

                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM plugin_fbamp_feedback_context"
                ).fetchone()
                total_feedback = row["cnt"] if row else 0

                rows = conn.execute("""
                    SELECT skill_id, ema_score FROM plugin_fbamp_skill_scores
                """).fetchall()

                if rows:
                    total_score = sum(r["ema_score"] for r in rows)
                    avg_score = round(total_score / len(rows), 3)
                    for r in rows:
                        if r["ema_score"] > 1.1:
                            skills_boosted += 1
                        elif r["ema_score"] < 0.9:
                            skills_decayed += 1

        except sqlite3.Error:
            pass

    return JSONResponse({
        "total_feedback": total_feedback,
        "skills_boosted": skills_boosted,
        "skills_decayed": skills_decayed,
        "avg_score": avg_score,
        "decay_half_life_days": decay_half_life,
    })


@router.post("/api/recompute")
def recompute_scores():
    if not SCRIPT_PATH.exists():
        raise HTTPException(500, "compute_scores.py script not found")

    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "-v"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return JSONResponse({
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        })
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Recompute timed out") from None
    except Exception as e:
        raise HTTPException(500, f"Recompute failed: {e}") from e
