"""SQLite-backed store for skills, embeddings, and usage feedback.

Schema
------
skills       — indexed skill metadata + full content
embeddings   — float vectors (JSON) per skill
feedback     — (query_vector, skill_id, helpful) rows for boost calculation
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

    def get_skill_content(self, skill_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT content FROM skills WHERE id = ?", (skill_id,)
        ).fetchone()
        return row["content"] if row else None

    def close(self) -> None:
        self._conn.close()
