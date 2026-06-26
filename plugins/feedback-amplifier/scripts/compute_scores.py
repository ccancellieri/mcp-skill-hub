#!/usr/bin/env python3
"""compute_scores.py — Recompute all skill scores with EMA + decay.

This script is intended to be run periodically (e.g., via scheduled task) to:
1. Apply decay to skills that haven't been used recently
2. Recalculate EMA scores based on accumulated feedback
3. Clean up stale feedback_context entries

Decay formula:
    decayed_score = old_score * exp(-days_since_use / half_life)

EMA formula:
    new_score = old_score * (1 - alpha) + feedback_signal * alpha

Where feedback_signal is derived from the used_count / injection_count ratio.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

HUB_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = HUB_ROOT / "skill_hub.db"

DEFAULT_HALF_LIFE_DAYS = 30
DEFAULT_EMA_ALPHA = 0.15
MIN_INJECTIONS_FOR_DECAY = 3

DEBUG_LOG = Path.home() / ".claude" / "mcp-skill-hub" / "logs" / "fbamp-scores.log"


def _debug(msg: str) -> None:
    try:
        DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(DEBUG_LOG, "a") as f:
            f.write(f"[{ts}] {msg}\n")
    except OSError:
        pass


def _load_config(conn: sqlite3.Connection) -> dict:
    try:
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


def compute_decay_factor(days_since_use: float, half_life_days: float) -> float:
    """Compute exponential decay factor."""
    if days_since_use <= 0:
        return 1.0
    return math.exp(-days_since_use / half_life_days)


def main(dry_run: bool = False, verbose: bool = False) -> int:
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=rwc", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as e:
        print(f"Error: failed to connect to database: {e}", file=sys.stderr)
        return 1

    try:
        config = _load_config(conn)
        half_life = float(config.get("decay_half_life_days", DEFAULT_HALF_LIFE_DAYS))
        ema_alpha = float(config.get("ema_alpha", DEFAULT_EMA_ALPHA))
        min_injections = int(config.get("min_injections_for_decay", MIN_INJECTIONS_FOR_DECAY))

        scores = conn.execute("""
            SELECT skill_id, ema_score, last_used_at, injection_count, used_count
            FROM plugin_fbamp_skill_scores
        """).fetchall()

        if not scores:
            if verbose:
                print("No skill scores to process.")
            return 0

        now = datetime.now()
        updates = []

        for row in scores:
            skill_id = row["skill_id"]
            old_score = float(row["ema_score"])
            last_used = row["last_used_at"]
            injection_count = int(row["injection_count"] or 0)
            used_count = int(row["used_count"] or 0)

            decay_factor = 1.0
            if injection_count >= min_injections and last_used:
                try:
                    last_dt = datetime.fromisoformat(last_used)
                    days_since = (now - last_dt).days
                    decay_factor = compute_decay_factor(days_since, half_life)
                except (ValueError, TypeError):
                    pass

            if injection_count > 0:
                usage_ratio = used_count / injection_count
            else:
                usage_ratio = 0.5

            feedback_signal = 0.5 + (usage_ratio - 0.5) * 2.0
            feedback_signal = max(0.3, min(2.0, feedback_signal))

            decayed_score = old_score * decay_factor
            new_score = decayed_score * (1 - ema_alpha) + feedback_signal * ema_alpha
            new_score = round(max(0.3, min(2.0, new_score)), 4)

            updates.append({
                "skill_id": skill_id,
                "old_score": old_score,
                "new_score": new_score,
                "decay_factor": decay_factor,
                "usage_ratio": usage_ratio,
            })

        if verbose:
            print(f"Processing {len(updates)} skill scores...")
            for u in updates:
                change = u["new_score"] - u["old_score"]
                direction = "↑" if change > 0.001 else "↓" if change < -0.001 else "="
                print(f"  {u['skill_id']}: {u['old_score']:.3f} → {u['new_score']:.3f} {direction} "
                      f"(decay={u['decay_factor']:.2f}, usage={u['usage_ratio']:.0%})")

        if not dry_run:
            for u in updates:
                conn.execute("""
                    UPDATE plugin_fbamp_skill_scores
                    SET ema_score = ?, decay_applied_at = datetime('now'), updated_at = datetime('now')
                    WHERE skill_id = ?
                """, (u["new_score"], u["skill_id"]))

            conn.execute("""
                DELETE FROM plugin_fbamp_feedback_context
                WHERE created_at < datetime('now', '-90 days')
            """)

            conn.commit()
            _debug(f"Updated {len(updates)} skill scores, cleaned stale entries")

            for u in updates:
                if abs(u["new_score"] - u["old_score"]) > 0.1:
                    target_val = 1.5 if u["new_score"] > u["old_score"] else 0.5
                    conn.execute("""
                        UPDATE skills
                        SET feedback_score = ROUND(
                            COALESCE(feedback_score, 1.0) * 0.85 + ? * 0.15,
                            4
                        )
                        WHERE id = ?
                    """, (target_val, u["skill_id"]))
            conn.commit()

        else:
            print(f"DRY RUN: would update {len(updates)} skill scores")

    except sqlite3.Error as e:
        print(f"Error: database operation failed: {e}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute skill scores with EMA + decay")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed output")
    args = parser.parse_args()
    sys.exit(main(dry_run=args.dry_run, verbose=args.verbose))
