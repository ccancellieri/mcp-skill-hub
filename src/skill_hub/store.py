"""SQLite-backed store for skills, embeddings, usage feedback, teachings, and tasks.

Schema
------
skills       — indexed skill metadata + full content
embeddings   — float vectors (JSON) per skill
feedback     — (query_vector, skill_id, helpful) rows for boost calculation
teachings    — explicit user rules ("when X, suggest Y")
plugins      — plugin metadata + embedded descriptions for suggestion
session_log  — automatic per-session tool usage for passive learning
tasks        — saved/closed conversation digests for cross-session context
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

            CREATE TABLE IF NOT EXISTS tasks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT NOT NULL,
                summary     TEXT NOT NULL,           -- raw or compacted summary
                context     TEXT,                    -- extra context, plans, decisions
                status      TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed')),
                tags        TEXT,                    -- comma-separated tags
                compact     TEXT,                    -- LLM-compacted digest (on close)
                vector      TEXT,                    -- JSON embedding of summary
                session_id  TEXT,                    -- which session created it
                created_at  TEXT DEFAULT (datetime('now')),
                updated_at  TEXT DEFAULT (datetime('now')),
                closed_at   TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status);

            CREATE TABLE IF NOT EXISTS interceptions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                command_type    TEXT NOT NULL,   -- "save_task", "close_task", "list_tasks", "search_context"
                message_preview TEXT,            -- first 100 chars of intercepted message
                estimated_tokens INTEGER NOT NULL DEFAULT 0,
                created_at      TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_interceptions_type ON interceptions (command_type);

            CREATE TABLE IF NOT EXISTS conversation_state (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT,
                message_count   INTEGER NOT NULL DEFAULT 0,
                digest          TEXT,           -- JSON conversation digest from local LLM
                stale_topics    TEXT,           -- JSON array of topics no longer active
                suggested_profile TEXT,
                created_at      TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS triage_log (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                message_preview     TEXT,
                action              TEXT NOT NULL,   -- local_answer, local_action, enrich_and_forward, pass_through
                confidence          REAL NOT NULL DEFAULT 0.0,
                estimated_tokens_saved INTEGER NOT NULL DEFAULT 0,
                created_at          TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_triage_action ON triage_log (action);

            CREATE TABLE IF NOT EXISTS context_injections (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                message_preview TEXT,
                skills_found    INTEGER NOT NULL DEFAULT 0,
                tasks_found     INTEGER NOT NULL DEFAULT 0,
                teachings_found INTEGER NOT NULL DEFAULT 0,
                memory_found    INTEGER NOT NULL DEFAULT 0,
                precompacted    INTEGER NOT NULL DEFAULT 0,
                chars_injected  INTEGER NOT NULL DEFAULT 0,
                created_at      TEXT DEFAULT (datetime('now'))
            );
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

    # ------------------------------------------------------------------
    # Tasks (conversation digests)

    def save_task(self, title: str, summary: str, vector: list[float],
                  context: str = "", tags: str = "",
                  session_id: str = "") -> int:
        cur = self._conn.execute("""
            INSERT INTO tasks (title, summary, context, tags, vector, session_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (title, summary, context, tags, json.dumps(vector), session_id))
        self._conn.commit()
        return cur.lastrowid or 0

    def update_task(self, task_id: int, summary: str = "",
                    context: str = "", tags: str = "",
                    vector: list[float] | None = None) -> bool:
        parts: list[str] = ["updated_at = datetime('now')"]
        params: list = []
        if summary:
            parts.append("summary = ?")
            params.append(summary)
        if context:
            parts.append("context = ?")
            params.append(context)
        if tags:
            parts.append("tags = ?")
            params.append(tags)
        if vector is not None:
            parts.append("vector = ?")
            params.append(json.dumps(vector))
        params.append(task_id)
        cur = self._conn.execute(
            f"UPDATE tasks SET {', '.join(parts)} WHERE id = ?", params
        )
        self._conn.commit()
        return cur.rowcount > 0

    def close_task(self, task_id: int, compact: str,
                   compact_vector: list[float] | None = None) -> bool:
        params: list = [compact]
        vec_clause = ""
        if compact_vector is not None:
            vec_clause = ", vector = ?"
            params.append(json.dumps(compact_vector))
        params.append(task_id)
        cur = self._conn.execute(f"""
            UPDATE tasks
            SET status = 'closed', compact = ?, closed_at = datetime('now'),
                updated_at = datetime('now'){vec_clause}
            WHERE id = ? AND status = 'open'
        """, params)
        self._conn.commit()
        return cur.rowcount > 0

    def reopen_task(self, task_id: int) -> bool:
        cur = self._conn.execute("""
            UPDATE tasks SET status = 'open', closed_at = NULL,
                             updated_at = datetime('now')
            WHERE id = ?
        """, (task_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def list_tasks(self, status: str = "open") -> list[sqlite3.Row]:
        if status == "all":
            return self._conn.execute(
                "SELECT id, title, summary, status, tags, created_at, updated_at, closed_at "
                "FROM tasks ORDER BY updated_at DESC"
            ).fetchall()
        return self._conn.execute(
            "SELECT id, title, summary, status, tags, created_at, updated_at, closed_at "
            "FROM tasks WHERE status = ? ORDER BY updated_at DESC",
            (status,)
        ).fetchall()

    def get_task(self, task_id: int) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()

    def search_tasks(self, query_vector: list[float], top_k: int = 3,
                     status: str = "all", min_sim: float = 0.4) -> list[dict]:
        """Search tasks by semantic similarity."""
        where = "" if status == "all" else f"AND t.status = '{status}'"
        rows = self._conn.execute(f"""
            SELECT t.id, t.title, t.summary, t.context, t.status,
                   t.tags, t.compact, t.vector, t.created_at, t.closed_at
            FROM tasks t
            WHERE t.vector IS NOT NULL {where}
        """).fetchall()

        scored: list[tuple[float, dict]] = []
        for row in rows:
            vec = json.loads(row["vector"])
            sim = _cosine(query_vector, vec)
            if sim >= min_sim:
                d = dict(row)
                d["similarity"] = sim
                scored.append((sim, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def get_skill_content(self, skill_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT content FROM skills WHERE id = ?", (skill_id,)
        ).fetchone()
        return row["content"] if row else None

    # ------------------------------------------------------------------
    # Token profiling (hook interception stats)

    def log_interception(self, command_type: str, message_preview: str,
                         estimated_tokens: int) -> None:
        self._conn.execute("""
            INSERT INTO interceptions (command_type, message_preview, estimated_tokens)
            VALUES (?, ?, ?)
        """, (command_type, message_preview[:100], estimated_tokens))
        self._conn.commit()

    def get_interception_stats(self) -> list[sqlite3.Row]:
        return self._conn.execute("""
            SELECT command_type,
                   COUNT(*) as intercept_count,
                   SUM(estimated_tokens) as total_tokens_saved
            FROM interceptions
            GROUP BY command_type
            ORDER BY total_tokens_saved DESC
        """).fetchall()

    def get_interception_totals(self) -> sqlite3.Row | None:
        return self._conn.execute("""
            SELECT COUNT(*) as total_interceptions,
                   SUM(estimated_tokens) as total_tokens_saved
            FROM interceptions
        """).fetchone()

    # ------------------------------------------------------------------
    # Context injection stats

    def log_context_injection(self, message_preview: str, skills: int,
                              tasks: int, teachings: int, memory: int,
                              precompacted: bool, chars: int) -> None:
        self._conn.execute("""
            INSERT INTO context_injections
                (message_preview, skills_found, tasks_found, teachings_found,
                 memory_found, precompacted, chars_injected)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (message_preview[:100], skills, tasks, teachings, memory,
              1 if precompacted else 0, chars))
        self._conn.commit()

    def get_context_injection_stats(self) -> dict:
        row = self._conn.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN skills_found > 0 THEN 1 ELSE 0 END) as with_skills,
                   SUM(CASE WHEN tasks_found > 0 THEN 1 ELSE 0 END) as with_tasks,
                   SUM(CASE WHEN teachings_found > 0 THEN 1 ELSE 0 END) as with_teachings,
                   SUM(CASE WHEN memory_found > 0 THEN 1 ELSE 0 END) as with_memory,
                   SUM(precompacted) as precompacted,
                   SUM(chars_injected) as total_chars,
                   AVG(chars_injected) as avg_chars
            FROM context_injections
        """).fetchone()
        return dict(row) if row else {}

    # ------------------------------------------------------------------
    # Triage stats

    def log_triage(self, message_preview: str, action: str,
                   confidence: float, estimated_tokens_saved: int) -> None:
        self._conn.execute("""
            INSERT INTO triage_log
                (message_preview, action, confidence, estimated_tokens_saved)
            VALUES (?, ?, ?, ?)
        """, (message_preview[:100], action, confidence, estimated_tokens_saved))
        self._conn.commit()

    def get_triage_stats(self) -> dict:
        row = self._conn.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN action = 'local_answer' THEN 1 ELSE 0 END) as local_answers,
                   SUM(CASE WHEN action = 'local_action' THEN 1 ELSE 0 END) as local_actions,
                   SUM(CASE WHEN action = 'enrich_and_forward' THEN 1 ELSE 0 END) as enriched,
                   SUM(CASE WHEN action = 'pass_through' THEN 1 ELSE 0 END) as passed,
                   SUM(estimated_tokens_saved) as total_tokens_saved,
                   AVG(confidence) as avg_confidence
            FROM triage_log
        """).fetchone()
        return dict(row) if row else {}

    # ------------------------------------------------------------------
    # Conversation state tracking

    def save_conversation_state(self, session_id: str, message_count: int,
                                digest: str, stale_topics: str,
                                suggested_profile: str | None) -> int:
        cur = self._conn.execute("""
            INSERT INTO conversation_state
                (session_id, message_count, digest, stale_topics, suggested_profile)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, message_count, digest, stale_topics, suggested_profile))
        self._conn.commit()
        return cur.lastrowid or 0

    def get_latest_conversation_state(self, session_id: str) -> sqlite3.Row | None:
        return self._conn.execute("""
            SELECT * FROM conversation_state
            WHERE session_id = ?
            ORDER BY id DESC LIMIT 1
        """, (session_id,)).fetchone()

    def get_message_count(self, session_id: str) -> int:
        """Get latest message count for a session."""
        row = self._conn.execute("""
            SELECT message_count FROM conversation_state
            WHERE session_id = ?
            ORDER BY id DESC LIMIT 1
        """, (session_id,)).fetchone()
        return row["message_count"] if row else 0

    # ------------------------------------------------------------------
    # Aggregate counts (for activity log banner)

    def count_skills(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) as n FROM skills").fetchone()
        return row["n"] if row else 0

    def count_tasks(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) as n FROM tasks GROUP BY status"
        ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    def count_teachings(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) as n FROM teachings").fetchone()
        return row["n"] if row else 0

    def count_interceptions(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) as n FROM interceptions").fetchone()
        return row["n"] if row else 0

    def total_tokens_saved(self) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(estimated_tokens), 0) as n FROM interceptions"
        ).fetchone()
        return row["n"] if row else 0

    def close(self) -> None:
        self._conn.close()
