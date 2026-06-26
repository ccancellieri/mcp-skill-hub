#!/usr/bin/env python3
"""on_skill_activated hook — capture context when a skill is suggested/injected.

This hook is dispatched by the skill-hub core when a skill is loaded for
injection into the system prompt. We capture the query, domain hints, and
session context so we can later correlate with whether the skill was actually
used (skill.used event) or ignored.

Payload (from plugin_hooks.dispatch):
    {
        "event": "on_skill_activated",
        "skill_id": "ogc-api-standards",
        "query": "implement OGC API Features endpoint",
        "domain_hints": ["ogc", "api", "standards"],
        "session_id": "abc123",
        "injection_id": 42   // optional, if known
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


def _debug(msg: str) -> None:
    try:
        DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(DEBUG_LOG, "a") as f:
            f.write(f"{msg}\n")
    except OSError:
        pass


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return 0

    event = payload.get("event", "")
    if event != "on_skill_activated":
        return 0

    skill_id = payload.get("skill_id", "")
    session_id = payload.get("session_id", "")
    query = payload.get("query", "")
    domain_hints = payload.get("domain_hints", [])
    injection_id = payload.get("injection_id")

    if not skill_id or not session_id:
        return 0

    _debug(f"on_skill_activated: skill={skill_id} session={session_id[:12]}")

    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=rwc", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as e:
        _debug(f"db connect failed: {e}")
        return 0

    try:
        conn.execute("""
            INSERT INTO plugin_fbamp_feedback_context
                (skill_id, session_id, query, domain_hints, injection_id, ts)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            skill_id,
            session_id,
            query[:500] if query else "",
            json.dumps(domain_hints) if domain_hints else "[]",
            injection_id,
            time.time(),
        ))
        conn.commit()

        conn.execute("""
            INSERT INTO plugin_fbamp_skill_scores (skill_id, injection_count, updated_at)
            VALUES (?, 1, datetime('now'))
            ON CONFLICT(skill_id) DO UPDATE SET
                injection_count = injection_count + 1,
                updated_at = datetime('now')
        """, (skill_id,))
        conn.commit()

    except sqlite3.Error as e:
        _debug(f"db write failed: {e}")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
