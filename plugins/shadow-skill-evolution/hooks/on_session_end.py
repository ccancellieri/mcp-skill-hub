#!/usr/bin/env python3
"""on_session_end hook for shadow-skill-evolution plugin.

Extracts tool-call patterns from the session's event log and stores them
as tool chains for later clustering and skill proposal generation.

Payload (from plugin_hooks.dispatch):
    - session_id: The session that just ended
    - topic: The session's tracked topic (if any)
    - summary: The session summary text
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path.home() / ".claude" / "mcp-skill-hub" / "skill_hub.db"
PLUGIN_DB_PATH = Path.home() / ".claude" / "mcp-skill-hub" / "plugins" / "shadow_skill_evolution.db"


def _get_store_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _get_plugin_conn() -> sqlite3.Connection:
    PLUGIN_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(PLUGIN_DB_PATH))
    conn.row_factory = sqlite3.Row
    schema_path = PLUGIN_ROOT / "storage" / "schema.sql"
    if schema_path.exists():
        conn.executescript(schema_path.read_text())
        conn.commit()
    return conn


def _hash_args(args: dict) -> str:
    normalized = json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha1(normalized.encode()).hexdigest()[:8]


def _extract_tool_chain(store_conn: sqlite3.Connection, session_id: str) -> list[tuple[str, str]]:
    events = store_conn.execute(
        "SELECT tool_name, payload FROM events "
        "WHERE session_id = ? AND kind = 'tool_invoke' "
        "ORDER BY ts ASC",
        (session_id,),
    ).fetchall()

    chain = []
    for event in events:
        tool_name = event["tool_name"]
        if not tool_name:
            continue
        try:
            payload = json.loads(event["payload"]) if event["payload"] else {}
        except json.JSONDecodeError:
            payload = {}

        args_hash = _hash_args(payload.get("args", payload))
        chain.append((tool_name, args_hash))

    return chain


def _compute_chain_hash(chain: list[tuple[str, str]]) -> str:
    chain_str = " → ".join(f"{t}:{h}" for t, h in chain)
    return hashlib.sha1(chain_str.encode()).hexdigest()[:16]


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return 0

    session_id = payload.get("session_id", "")
    topic = payload.get("topic", "")[:200]
    summary = payload.get("summary", "")[:500]

    if not session_id:
        return 0

    try:
        store_conn = _get_store_conn()
        plugin_conn = _get_plugin_conn()
    except sqlite3.Error:
        return 0

    try:
        chain = _extract_tool_chain(store_conn, session_id)
        if len(chain) < 2:
            return 0

        chain_hash = _compute_chain_hash(chain)
        tool_sequence = json.dumps(chain)

        existing = plugin_conn.execute(
            "SELECT id, occurrence_count FROM tool_chains "
            "WHERE chain_hash = ? AND session_id = ?",
            (chain_hash, session_id),
        ).fetchone()

        if existing:
            plugin_conn.execute(
                "UPDATE tool_chains SET occurrence_count = ?, last_seen_at = datetime('now') "
                "WHERE id = ?",
                (existing["occurrence_count"] + 1, existing["id"]),
            )
        else:
            metadata = json.dumps({"topic": topic, "summary": summary})
            plugin_conn.execute(
                "INSERT INTO tool_chains (session_id, chain_hash, tool_sequence, metadata) "
                "VALUES (?, ?, ?, ?)",
                (session_id, chain_hash, tool_sequence, metadata),
            )

        plugin_conn.commit()
    except sqlite3.Error:
        pass
    finally:
        store_conn.close()
        plugin_conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
