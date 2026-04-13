"""Shared sqlite-backed verdict cache for the auto-approve hooks.

Used by both ``auto_approve.py`` (PreToolUse; reads cache, optional LLM query)
and ``post_tool_observer.py`` (PostToolUse; writes "user_approved" verdicts).

Kept stdlib-only so hooks start fast — no skill_hub import.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from pathlib import Path

DB_PATH = Path.home() / ".claude" / "mcp-skill-hub" / "command_verdicts.db"
CONFIG_PATH = Path.home() / ".claude" / "mcp-skill-hub" / "config.json"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS command_verdicts (
    cmd_hash      TEXT PRIMARY KEY,
    tool_name     TEXT NOT NULL,
    command       TEXT NOT NULL,
    decision      TEXT NOT NULL,        -- allow | deny
    source        TEXT NOT NULL,        -- user_approved | llm | static
    confidence    REAL NOT NULL DEFAULT 1.0,
    hit_count     INTEGER NOT NULL DEFAULT 0,
    created_at    INTEGER NOT NULL,
    last_used_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_verdict_recent
  ON command_verdicts (last_used_at DESC);
"""


def load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _canon(tool_name: str, command: str) -> str:
    """Normalise a command so trivially-different invocations share a cache key.

    Replaces absolute paths, uuids, hex, and numbers with placeholders so
    ``git log -n 20`` and ``git log -n 100`` hash the same. Tool name keeps
    them segregated (Bash vs Write vs Edit).
    """
    s = command.strip()
    s = re.sub(r"/[^\s'\"]+", "/PATH", s)        # absolute paths
    s = re.sub(r"\b[0-9a-f]{7,40}\b", "HEX", s)  # commit shas / hashes
    s = re.sub(r"\b\d+\b", "N", s)               # numbers
    s = re.sub(r"\s+", " ", s)
    return f"{tool_name}\x1f{s}"


def hash_key(tool_name: str, command: str) -> str:
    return hashlib.sha1(_canon(tool_name, command).encode()).hexdigest()


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """Idempotent migration: add vector + pinned columns if missing."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(command_verdicts)")}
    if "vector" not in cols:
        conn.execute("ALTER TABLE command_verdicts ADD COLUMN vector TEXT")
    if "pinned" not in cols:
        conn.execute(
            "ALTER TABLE command_verdicts ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0"
        )
    conn.commit()


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=2.0)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    _ensure_columns(conn)
    return conn


def lookup(conn: sqlite3.Connection, tool_name: str, command: str,
           ttl_days: int = 30) -> dict | None:
    key = hash_key(tool_name, command)
    row = conn.execute(
        "SELECT * FROM command_verdicts WHERE cmd_hash = ?", (key,)
    ).fetchone()
    if not row:
        return None
    age_days = (time.time() - row["created_at"]) / 86400.0
    if age_days > ttl_days:
        return None
    conn.execute(
        "UPDATE command_verdicts SET hit_count = hit_count + 1, "
        "last_used_at = ? WHERE cmd_hash = ?",
        (int(time.time()), key),
    )
    conn.commit()
    return dict(row)


def put(conn: sqlite3.Connection, tool_name: str, command: str,
        decision: str, source: str, confidence: float = 1.0) -> None:
    key = hash_key(tool_name, command)
    now = int(time.time())
    # Priority: user_approved > llm > static (don't let llm overwrite user_approved)
    priority = {"static": 0, "llm": 1, "user_approved": 2}
    existing = conn.execute(
        "SELECT source FROM command_verdicts WHERE cmd_hash = ?", (key,)
    ).fetchone()
    if existing and priority.get(source, 0) < priority.get(existing["source"], 0):
        return
    conn.execute(
        "INSERT INTO command_verdicts "
        "  (cmd_hash, tool_name, command, decision, source, confidence, "
        "   created_at, last_used_at) "
        "VALUES (?,?,?,?,?,?,?,?) "
        "ON CONFLICT(cmd_hash) DO UPDATE SET "
        "  decision=excluded.decision, source=excluded.source, "
        "  confidence=excluded.confidence, last_used_at=excluded.last_used_at",
        (key, tool_name, command[:2000], decision, source, confidence, now, now),
    )
    conn.commit()


def delete(conn: sqlite3.Connection, cmd_hash: str) -> bool:
    cur = conn.execute("DELETE FROM command_verdicts WHERE cmd_hash = ?", (cmd_hash,))
    conn.commit()
    return cur.rowcount > 0


def set_pinned(conn: sqlite3.Connection, cmd_hash: str, pinned: bool) -> bool:
    cur = conn.execute(
        "UPDATE command_verdicts SET pinned = ? WHERE cmd_hash = ?",
        (1 if pinned else 0, cmd_hash),
    )
    conn.commit()
    return cur.rowcount > 0


def update_decision(conn: sqlite3.Connection, cmd_hash: str, decision: str) -> bool:
    cur = conn.execute(
        "UPDATE command_verdicts SET decision = ? WHERE cmd_hash = ?",
        (decision, cmd_hash),
    )
    conn.commit()
    return cur.rowcount > 0


def put_with_vector(conn: sqlite3.Connection, tool_name: str, command: str,
                    decision: str, source: str, vector: list[float] | None,
                    confidence: float = 1.0) -> None:
    """Like put(), but also stores the embedding vector (JSON)."""
    key = hash_key(tool_name, command)
    now = int(time.time())
    priority = {"static": 0, "claude_session": 1, "llm": 1,
                "vector": 1, "user_approved": 2}
    existing = conn.execute(
        "SELECT source, pinned FROM command_verdicts WHERE cmd_hash = ?", (key,)
    ).fetchone()
    if existing:
        if existing["pinned"]:
            return
        if priority.get(source, 0) < priority.get(existing["source"], 0):
            return
    vec_json = json.dumps(vector) if vector else None
    conn.execute(
        "INSERT INTO command_verdicts "
        "  (cmd_hash, tool_name, command, decision, source, confidence, "
        "   created_at, last_used_at, vector) "
        "VALUES (?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(cmd_hash) DO UPDATE SET "
        "  decision=excluded.decision, source=excluded.source, "
        "  confidence=excluded.confidence, last_used_at=excluded.last_used_at, "
        "  vector=COALESCE(excluded.vector, command_verdicts.vector)",
        (key, tool_name, command[:2000], decision, source, confidence,
         now, now, vec_json),
    )
    conn.commit()


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def search_by_vector(conn: sqlite3.Connection, vector: list[float],
                     threshold: float = 0.88,
                     source_filter: str = "user_approved") -> dict | None:
    """Cosine search over stored vectors. Returns best match above threshold or None."""
    if not vector:
        return None
    query = (
        "SELECT * FROM command_verdicts "
        "WHERE vector IS NOT NULL AND decision='allow' "
    )
    params: list = []
    if source_filter:
        query += "AND source = ? "
        params.append(source_filter)
    rows = conn.execute(query, params).fetchall()
    best_sim = 0.0
    best_row = None
    for row in rows:
        try:
            vec = json.loads(row["vector"])
        except (json.JSONDecodeError, TypeError):
            continue
        sim = _cosine(vector, vec)
        if sim > best_sim:
            best_sim = sim
            best_row = row
    if best_row and best_sim >= threshold:
        d = dict(best_row)
        d["similarity"] = best_sim
        return d
    return None


def recent_examples(conn: sqlite3.Connection, n: int = 6) -> list[dict]:
    """Fetch the most recent user-approved entries for few-shot LLM prompts."""
    rows = conn.execute(
        "SELECT tool_name, command, decision FROM command_verdicts "
        "WHERE source = 'user_approved' "
        "ORDER BY last_used_at DESC LIMIT ?",
        (n,),
    ).fetchall()
    return [dict(r) for r in rows]
