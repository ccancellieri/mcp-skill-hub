"""SQLite-backed store for skills, embeddings, usage feedback, and teachings.

Schema
------
skills       — indexed skill metadata + full content
embeddings   — float vectors (JSON) per skill
feedback     — (query_vector, skill_id, helpful) rows for boost calculation
teachings    — explicit user rules ("when X, suggest Y")
plugins      — plugin metadata + embedded descriptions for suggestion
session_log  — automatic per-session tool usage for passive learning
"""

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

DB_PATH = Path.home() / ".claude" / "mcp-skill-hub" / "skill_hub.db"


@dataclass
class Skill:
    id: str          # e.g. "superpowers:brainstorm"
    name: str
    description: str
    content: str
    file_path: str
    plugin: str


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


class SkillStore:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._migrate()

    def _migrate(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS skills (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                description TEXT,
                content     TEXT NOT NULL,
                file_path   TEXT,
                plugin      TEXT,
                indexed_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                skill_id    TEXT PRIMARY KEY REFERENCES skills(id) ON DELETE CASCADE,
                model       TEXT NOT NULL,
                vector      TEXT NOT NULL   -- JSON array of floats
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                query        TEXT NOT NULL,
                query_vector TEXT,           -- JSON array, for similar-query boost
                skill_id     TEXT NOT NULL,
                helpful      INTEGER NOT NULL CHECK (helpful IN (0, 1)),
                created_at   TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_feedback_skill ON feedback (skill_id);

            CREATE TABLE IF NOT EXISTS teachings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                rule        TEXT NOT NULL,          -- "when I give a URL..."
                rule_vector TEXT NOT NULL,           -- JSON embedding of rule
                action      TEXT NOT NULL,           -- "suggest chrome-devtools-mcp"
                target_type TEXT NOT NULL DEFAULT 'plugin',  -- 'plugin' or 'skill'
                target_id   TEXT NOT NULL,           -- plugin or skill id
                weight      REAL NOT NULL DEFAULT 1.0,
                created_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS plugins (
                id          TEXT PRIMARY KEY,        -- e.g. "chrome-devtools-mcp@claude-plugins-official"
                short_name  TEXT NOT NULL,           -- e.g. "chrome-devtools-mcp"
                description TEXT,
                created_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS plugin_embeddings (
                plugin_id   TEXT PRIMARY KEY REFERENCES plugins(id) ON DELETE CASCADE,
                model       TEXT NOT NULL,
                vector      TEXT NOT NULL            -- JSON array of floats
            );

            CREATE TABLE IF NOT EXISTS session_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,           -- unique per session
                query       TEXT,                    -- first user message / topic
                query_vector TEXT,                   -- embedded query
                tool_used   TEXT NOT NULL,           -- MCP tool name that was called
                plugin_id   TEXT,                    -- resolved plugin
                created_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_session_log_plugin ON session_log (plugin_id);
            CREATE INDEX IF NOT EXISTS idx_session_log_session ON session_log (session_id);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write

    def upsert_skill(self, skill: Skill) -> None:
        self._conn.execute("""
            INSERT INTO skills (id, name, description, content, file_path, plugin)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name        = excluded.name,
                description = excluded.description,
                content     = excluded.content,
                file_path   = excluded.file_path,
                plugin      = excluded.plugin,
                indexed_at  = datetime('now')
        """, (skill.id, skill.name, skill.description, skill.content,
              skill.file_path, skill.plugin))
        self._conn.commit()

    def upsert_embedding(self, skill_id: str, model: str, vector: list[float]) -> None:
        self._conn.execute("""
            INSERT INTO embeddings (skill_id, model, vector)
            VALUES (?, ?, ?)
            ON CONFLICT(skill_id) DO UPDATE SET
                model  = excluded.model,
                vector = excluded.vector
        """, (skill_id, model, json.dumps(vector)))
        self._conn.commit()

    def record_feedback(self, skill_id: str, query: str,
                        query_vector: list[float], helpful: bool) -> None:
        self._conn.execute("""
            INSERT INTO feedback (query, query_vector, skill_id, helpful)
            VALUES (?, ?, ?, ?)
        """, (query, json.dumps(query_vector), skill_id, int(helpful)))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read

    def list_skills(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT id, name, description, plugin FROM skills ORDER BY name"
        ).fetchall()

    def search(self, query_vector: list[float], top_k: int = 3,
               similarity_threshold: float = 0.3) -> list[dict]:
        """Return top-k skills by cosine similarity, boosted by past feedback."""
        rows = self._conn.execute("""
            SELECT s.id, s.name, s.description, s.content, s.plugin,
                   e.vector, e.model
            FROM skills s
            JOIN embeddings e ON e.skill_id = s.id
        """).fetchall()

        scored: list[tuple[float, dict]] = []
        for row in rows:
            vec = json.loads(row["vector"])
            sim = _cosine(query_vector, vec)
            if sim < similarity_threshold:
                continue
            boost = self._feedback_boost(row["id"], query_vector)
            scored.append((sim * boost, dict(row)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def _feedback_boost(self, skill_id: str, query_vector: list[float],
                        min_sim: float = 0.75) -> float:
        """
        Boost factor in [1.0, 1.5] derived from past feedback on similar queries.
        Uses cosine similarity between current query and stored feedback query vectors.
        """
        rows = self._conn.execute(
            "SELECT query_vector, helpful FROM feedback WHERE skill_id = ?",
            (skill_id,)
        ).fetchall()

        positive = negative = 0.0
        for row in rows:
            past_vec = json.loads(row["query_vector"])
            sim = _cosine(query_vector, past_vec)
            if sim < min_sim:
                continue
            if row["helpful"]:
                positive += sim
            else:
                negative += sim

        total = positive + negative
        if total == 0.0:
            return 1.0
        return 1.0 + (positive / total) * 0.5  # max 1.5x boost

    # ------------------------------------------------------------------
    # Teachings

    def add_teaching(self, rule: str, rule_vector: list[float],
                     action: str, target_type: str, target_id: str,
                     weight: float = 1.0) -> int:
        cur = self._conn.execute("""
            INSERT INTO teachings (rule, rule_vector, action, target_type, target_id, weight)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (rule, json.dumps(rule_vector), action, target_type, target_id, weight))
        self._conn.commit()
        return cur.lastrowid or 0

    def search_teachings(self, query_vector: list[float],
                         min_sim: float = 0.6) -> list[dict]:
        """Find teachings whose rule matches the query semantically."""
        rows = self._conn.execute(
            "SELECT id, rule, rule_vector, action, target_type, target_id, weight "
            "FROM teachings"
        ).fetchall()

        results: list[tuple[float, dict]] = []
        for row in rows:
            vec = json.loads(row["rule_vector"])
            sim = _cosine(query_vector, vec)
            if sim >= min_sim:
                results.append((sim * row["weight"], dict(row)))

        results.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in results]

    def list_teachings(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT id, rule, action, target_type, target_id, weight FROM teachings "
            "ORDER BY created_at DESC"
        ).fetchall()

    def remove_teaching(self, teaching_id: int) -> bool:
        cur = self._conn.execute("DELETE FROM teachings WHERE id = ?", (teaching_id,))
        self._conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Plugins

    def upsert_plugin(self, plugin_id: str, short_name: str,
                      description: str) -> None:
        self._conn.execute("""
            INSERT INTO plugins (id, short_name, description)
            VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                short_name  = excluded.short_name,
                description = excluded.description
        """, (plugin_id, short_name, description))
        self._conn.commit()

    def upsert_plugin_embedding(self, plugin_id: str, model: str,
                                vector: list[float]) -> None:
        self._conn.execute("""
            INSERT INTO plugin_embeddings (plugin_id, model, vector)
            VALUES (?, ?, ?)
            ON CONFLICT(plugin_id) DO UPDATE SET
                model  = excluded.model,
                vector = excluded.vector
        """, (plugin_id, model, json.dumps(vector)))
        self._conn.commit()

    def suggest_plugins(self, query_vector: list[float],
                        min_sim: float = 0.4) -> list[dict]:
        """Suggest plugins by combining embedding similarity + teachings + session history."""
        # 1. Embedding-based similarity
        rows = self._conn.execute("""
            SELECT p.id, p.short_name, p.description, pe.vector
            FROM plugins p
            JOIN plugin_embeddings pe ON pe.plugin_id = p.id
        """).fetchall()

        scores: dict[str, dict] = {}
        for row in rows:
            vec = json.loads(row["vector"])
            sim = _cosine(query_vector, vec)
            if sim >= min_sim:
                scores[row["id"]] = {
                    "plugin_id": row["id"],
                    "short_name": row["short_name"],
                    "description": row["description"],
                    "embed_score": sim,
                    "teaching_score": 0.0,
                    "session_score": 0.0,
                }

        # 2. Teaching-based boost
        teachings = self.search_teachings(query_vector, min_sim=0.6)
        for t in teachings:
            if t["target_type"] == "plugin":
                pid = t["target_id"]
                if pid not in scores:
                    # Teaching overrides threshold — force include
                    plugin = self._conn.execute(
                        "SELECT id, short_name, description FROM plugins WHERE id = ? "
                        "OR short_name = ?", (pid, pid)
                    ).fetchone()
                    if plugin:
                        scores[plugin["id"]] = {
                            "plugin_id": plugin["id"],
                            "short_name": plugin["short_name"],
                            "description": plugin["description"],
                            "embed_score": 0.0,
                            "teaching_score": 0.0,
                            "session_score": 0.0,
                        }
                        pid = plugin["id"]
                if pid in scores:
                    scores[pid]["teaching_score"] += 0.3

        # 3. Session history boost
        history = self._conn.execute("""
            SELECT plugin_id, query_vector, COUNT(*) as cnt
            FROM session_log
            WHERE plugin_id IS NOT NULL
            GROUP BY plugin_id, query_vector
        """).fetchall()

        for row in history:
            if not row["query_vector"]:
                continue
            past_vec = json.loads(row["query_vector"])
            sim = _cosine(query_vector, past_vec)
            if sim >= 0.7 and row["plugin_id"] in scores:
                scores[row["plugin_id"]]["session_score"] += sim * 0.2 * row["cnt"]

        # Combine and rank
        ranked: list[tuple[float, dict]] = []
        for info in scores.values():
            total = info["embed_score"] + info["teaching_score"] + info["session_score"]
            info["total_score"] = total
            ranked.append((total, info))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in ranked]

    # ------------------------------------------------------------------
    # Session log

    def log_session_tool(self, session_id: str, query: str,
                         query_vector: list[float] | None,
                         tool_used: str, plugin_id: str | None) -> None:
        self._conn.execute("""
            INSERT INTO session_log (session_id, query, query_vector, tool_used, plugin_id)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, query,
              json.dumps(query_vector) if query_vector else None,
              tool_used, plugin_id))
        self._conn.commit()

    def get_session_stats(self, limit: int = 20) -> list[sqlite3.Row]:
        """Most-used plugins across recent sessions."""
        return self._conn.execute("""
            SELECT plugin_id, COUNT(*) as usage_count,
                   COUNT(DISTINCT session_id) as session_count
            FROM session_log
            WHERE plugin_id IS NOT NULL
            GROUP BY plugin_id
            ORDER BY usage_count DESC
            LIMIT ?
        """, (limit,)).fetchall()

    def get_skill_content(self, skill_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT content FROM skills WHERE id = ?", (skill_id,)
        ).fetchone()
        return row["content"] if row else None

    def close(self) -> None:
        self._conn.close()
