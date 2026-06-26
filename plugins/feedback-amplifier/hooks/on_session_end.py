#!/usr/bin/env python3
"""on_session_end hook — consolidate feedback and update skill scores.

This hook is dispatched once per session when it ends. We:

1. Query skill_injections for this session to see which skills were loaded
2. Query events for skill.used events to see which were actually used
3. Update plugin_fbamp_feedback_context.was_used accordingly
4. Apply EMA score updates based on usage patterns
5. Update domain-specific performance metrics

Payload (from plugin_hooks.dispatch via _cmd_session_close):
    {
        "event": "on_session_end",
        "session_id": "abc123",
        "topic": "Implementing OGC API",
        "summary": "Implemented /collections endpoint",
        "reason": "exit"
    }
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent
HUB_ROOT = PLUGIN_DIR.parent.parent
DB_PATH = HUB_ROOT / "skill_hub.db"

DEBUG_LOG = Path.home() / ".claude" / "mcp-skill-hub" / "logs" / "fbamp-hook.log"

DEFAULT_EMA_ALPHA = 0.15
BOOST_ON_USED = 0.1
PENALTY_NOT_USED = 0.05


def _debug(msg: str) -> None:
    try:
        DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(DEBUG_LOG, "a") as f:
            f.write(f"{msg}\n")
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


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return 0

    event = payload.get("event", "")
    if event != "on_session_end":
        return 0

    session_id = payload.get("session_id", "")
    if not session_id:
        return 0

    _debug(f"on_session_end: session={session_id[:12]}")

    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=rwc", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as e:
        _debug(f"db connect failed: {e}")
        return 0

    try:
        config = _load_config(conn)
        ema_alpha = float(config.get("ema_alpha", DEFAULT_EMA_ALPHA))
        boost = float(config.get("boost_on_skill_used", BOOST_ON_USED))
        penalty = float(config.get("penalty_on_injection_not_used", PENALTY_NOT_USED))

        injections = conn.execute(
            "SELECT id, skill_id, query FROM skill_injections WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        if not injections:
            _debug(f"no injections for session {session_id[:12]}")
            return 0

        injection_ids = [r["id"] for r in injections]
        injection_by_id = {r["id"]: r for r in injections}
        skills_injected = {r["skill_id"] for r in injections}

        used_events = conn.execute("""
            SELECT payload FROM events
            WHERE session_id = ? AND kind = 'skill.used'
        """, (session_id,)).fetchall()

        skills_used = set()
        for ev in used_events:
            try:
                data = json.loads(ev["payload"])
                sid = data.get("skill_id", "")
                if sid:
                    skills_used.add(sid)
            except (json.JSONDecodeError, TypeError):
                continue

        for skill_id in skills_injected:
            was_used = 1 if skill_id in skills_used else -1
            conn.execute("""
                UPDATE plugin_fbamp_feedback_context
                SET was_used = ?
                WHERE session_id = ? AND skill_id = ?
            """, (was_used, session_id, skill_id))

            if was_used == 1:
                conn.execute("""
                    UPDATE plugin_fbamp_skill_scores
                    SET used_count = used_count + 1,
                        last_used_at = datetime('now'),
                        updated_at = datetime('now')
                    WHERE skill_id = ?
                """, (skill_id,))

                row = conn.execute(
                    "SELECT ema_score FROM plugin_fbamp_skill_scores WHERE skill_id = ?",
                    (skill_id,),
                ).fetchone()
                if row:
                    old_score = float(row["ema_score"])
                    new_score = min(2.0, old_score + boost)
                    conn.execute(
                        "UPDATE plugin_fbamp_skill_scores SET ema_score = ? WHERE skill_id = ?",
                        (new_score, skill_id),
                    )

                _update_main_feedback_score(conn, skill_id, helpful=True)

            elif was_used == -1:
                row = conn.execute(
                    "SELECT ema_score FROM plugin_fbamp_skill_scores WHERE skill_id = ?",
                    (skill_id,),
                ).fetchone()
                if row:
                    old_score = float(row["ema_score"])
                    new_score = max(0.3, old_score - penalty)
                    conn.execute(
                        "UPDATE plugin_fbamp_skill_scores SET ema_score = ? WHERE skill_id = ?",
                        (new_score, skill_id),
                    )

                _update_main_feedback_score(conn, skill_id, helpful=False)

            _update_domain_performance(conn, skill_id, was_used == 1)

        conn.commit()
        _debug(f"session_end: {len(skills_injected)} skills, {len(skills_used)} used")

    except sqlite3.Error as e:
        _debug(f"db error: {e}")
    finally:
        conn.close()

    return 0


def _update_main_feedback_score(conn: sqlite3.Connection, skill_id: str, helpful: bool) -> None:
    """Update the main skills.feedback_score column to influence search ranking."""
    try:
        target_val = 1.5 if helpful else 0.5
        conn.execute("""
            UPDATE skills
            SET feedback_score = ROUND(
                COALESCE(feedback_score, 1.0) * 0.85 + ? * 0.15,
                4
            )
            WHERE id = ?
        """, (target_val, skill_id))
    except sqlite3.Error:
        pass


def _update_domain_performance(conn: sqlite3.Connection, skill_id: str, success: bool) -> None:
    """Update domain-specific performance metrics for the skill."""
    try:
        row = conn.execute("""
            SELECT domain_hints FROM plugin_fbamp_feedback_context
            WHERE skill_id = ?
            ORDER BY ts DESC LIMIT 1
        """, (skill_id,)).fetchone()

        if not row or not row["domain_hints"]:
            return

        try:
            domains = json.loads(row["domain_hints"])
        except (json.JSONDecodeError, TypeError):
            return

        if not isinstance(domains, list):
            return

        for domain in domains[:3]:
            if not domain or len(domain) < 2:
                continue
            domain = domain.lower().strip()[:50]

            if success:
                conn.execute("""
                    INSERT INTO plugin_fbamp_domain_performance
                        (skill_id, domain, success_count, total_count, last_at)
                    VALUES (?, ?, 1, 1, datetime('now'))
                    ON CONFLICT(skill_id, domain) DO UPDATE SET
                        success_count = success_count + 1,
                        total_count = total_count + 1,
                        last_at = datetime('now')
                """, (skill_id, domain))
            else:
                conn.execute("""
                    INSERT INTO plugin_fbamp_domain_performance
                        (skill_id, domain, success_count, total_count, last_at)
                    VALUES (?, ?, 0, 1, datetime('now'))
                    ON CONFLICT(skill_id, domain) DO UPDATE SET
                        total_count = total_count + 1,
                        last_at = datetime('now')
                """, (skill_id, domain))
    except sqlite3.Error:
        pass


if __name__ == "__main__":
    sys.exit(main())
