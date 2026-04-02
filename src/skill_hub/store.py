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
    target: str = "claude"  # "claude" or "local"


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
        # In-process vector cache: avoids JSON deserialization of all embedding
        # rows on every search() call.  Invalidated by upsert_embedding().
        # Structure: {skill_id: (vector_as_list, pre_stored_norm)}
        self._vec_cache: dict[str, tuple[list[float], float]] = {}
        self._vec_cache_valid: bool = False
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
                target      TEXT NOT NULL DEFAULT 'claude'
                                CHECK (target IN ('claude', 'local')),
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

            CREATE TABLE IF NOT EXISTS session_context (
                session_id      TEXT PRIMARY KEY,
                loaded_skills   TEXT NOT NULL DEFAULT '[]',   -- JSON array of skill IDs
                context_summary TEXT NOT NULL DEFAULT '',     -- rolling summary of conversation
                message_count   INTEGER NOT NULL DEFAULT 0,
                recent_messages TEXT NOT NULL DEFAULT '[]',   -- JSON array of last N user messages
                updated_at      TEXT DEFAULT (datetime('now'))
            );

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

        # Incremental migrations
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(skills)")}
        if "target" not in cols:
            self._conn.execute(
                "ALTER TABLE skills ADD COLUMN target TEXT NOT NULL DEFAULT 'claude'"
            )
            self._conn.commit()
        if "feedback_score" not in cols:
            # Pre-aggregated EMA feedback score: replaces per-search O(N) scan.
            # Updated incrementally in record_feedback(); read in search().
            self._conn.execute(
                "ALTER TABLE skills ADD COLUMN feedback_score REAL NOT NULL DEFAULT 1.0"
            )
            self._conn.commit()

        ctx_cols = {row[1] for row in self._conn.execute(
            "PRAGMA table_info(session_context)")}
        if "recent_messages" not in ctx_cols:
            self._conn.execute(
                "ALTER TABLE session_context ADD COLUMN "
                "recent_messages TEXT NOT NULL DEFAULT '[]'"
            )
            self._conn.commit()

        emb_cols = {row[1] for row in self._conn.execute("PRAGMA table_info(embeddings)")}
        if "norm" not in emb_cols:
            # Pre-stored L2 norm: avoids recomputing sqrt(sum(x²)) per skill per search.
            self._conn.execute(
                "ALTER TABLE embeddings ADD COLUMN norm REAL NOT NULL DEFAULT 0.0"
            )
            # Back-fill norms for existing rows
            rows = self._conn.execute(
                "SELECT skill_id, vector FROM embeddings WHERE norm = 0.0"
            ).fetchall()
            for row in rows:
                vec = json.loads(row["vector"])
                norm = math.sqrt(sum(x * x for x in vec))
                self._conn.execute(
                    "UPDATE embeddings SET norm = ? WHERE skill_id = ?",
                    (norm, row["skill_id"]),
                )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Write

    def upsert_skill(self, skill: Skill) -> None:
        self._conn.execute("""
            INSERT INTO skills (id, name, description, content, file_path, plugin, target)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name        = excluded.name,
                description = excluded.description,
                content     = excluded.content,
                file_path   = excluded.file_path,
                plugin      = excluded.plugin,
                target      = excluded.target,
                indexed_at  = datetime('now')
        """, (skill.id, skill.name, skill.description, skill.content,
              skill.file_path, skill.plugin, skill.target))
        self._conn.commit()

    def upsert_embedding(self, skill_id: str, model: str, vector: list[float]) -> None:
        norm = math.sqrt(sum(x * x for x in vector))
        self._conn.execute("""
            INSERT INTO embeddings (skill_id, model, vector, norm)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(skill_id) DO UPDATE SET
                model  = excluded.model,
                vector = excluded.vector,
                norm   = excluded.norm
        """, (skill_id, model, json.dumps(vector), norm))
        self._conn.commit()
        # Invalidate in-process vector cache so the next search reloads
        self._vec_cache_valid = False

    def record_feedback(self, skill_id: str, query: str,
                        query_vector: list[float], helpful: bool) -> None:
        self._conn.execute("""
            INSERT INTO feedback (query, query_vector, skill_id, helpful)
            VALUES (?, ?, ?, ?)
        """, (query, json.dumps(query_vector), skill_id, int(helpful)))
        # Update EMA feedback score on skills table:
        #   positive signal → nudge toward 1.5 (max boost)
        #   negative signal → nudge toward 0.5 (min boost)
        # EMA factor 0.15 keeps it responsive without thrashing on a single point.
        target_val = 1.5 if helpful else 0.5
        self._conn.execute("""
            UPDATE skills
            SET feedback_score = ROUND(
                COALESCE(feedback_score, 1.0) * 0.85 + ? * 0.15,
                4
            )
            WHERE id = ?
        """, (target_val, skill_id))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read

    def get_skill(self, skill_id: str) -> dict | None:
        """Get a single skill by its ID."""
        row = self._conn.execute(
            "SELECT id, name, description, content, plugin, target "
            "FROM skills WHERE id = ?", (skill_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_skills(self, target: str | None = None) -> list[sqlite3.Row]:
        if target:
            return self._conn.execute(
                "SELECT id, name, description, plugin, target FROM skills "
                "WHERE target = ? ORDER BY name", (target,)
            ).fetchall()
        return self._conn.execute(
            "SELECT id, name, description, plugin, target FROM skills ORDER BY name"
        ).fetchall()

    def _load_vec_cache(self) -> None:
        """Populate in-process vector cache from DB (once per process per index cycle)."""
        rows = self._conn.execute(
            "SELECT skill_id, vector, norm FROM embeddings"
        ).fetchall()
        self._vec_cache = {
            row["skill_id"]: (json.loads(row["vector"]), row["norm"] or 0.0)
            for row in rows
        }
        self._vec_cache_valid = True

    def search(self, query_vector: list[float], top_k: int = 3,
               similarity_threshold: float = 0.3,
               target: str | None = None) -> list[dict]:
        """Return top-k skills by cosine similarity, boosted by pre-aggregated feedback score.

        Uses an in-process vector cache to avoid JSON deserialization on every call.
        Uses pre-stored L2 norms to avoid recomputing sqrt per skill per search.
        """
        if not self._vec_cache_valid:
            self._load_vec_cache()

        # Query norm (computed once for this search)
        qnorm = math.sqrt(sum(x * x for x in query_vector))
        if qnorm == 0.0:
            return []

        # Fetch skill metadata + EMA feedback score (no vector column needed)
        if target:
            rows = self._conn.execute("""
                SELECT s.id, s.name, s.description, s.content, s.plugin,
                       s.target, s.feedback_score
                FROM skills s
                WHERE s.target = ?
            """, (target,)).fetchall()
        else:
            rows = self._conn.execute("""
                SELECT s.id, s.name, s.description, s.content, s.plugin,
                       s.target, s.feedback_score
                FROM skills s
            """).fetchall()

        scored: list[tuple[float, dict]] = []
        for row in rows:
            cached = self._vec_cache.get(row["id"])
            if cached is None:
                continue  # no embedding yet
            vec, snorm = cached
            if snorm == 0.0:
                continue
            # Cosine using pre-stored norm: avoids sqrt per skill
            dot = sum(a * b for a, b in zip(query_vector, vec))
            sim = dot / (qnorm * snorm)
            if sim < similarity_threshold:
                continue
            boost = float(row["feedback_score"] or 1.0)
            scored.append((sim * boost, dict(row)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]


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
    # Session context (dynamic skill lifecycle)

    def get_session_context(self, session_id: str) -> dict:
        """Get the current session context for dynamic skill management."""
        row = self._conn.execute(
            "SELECT * FROM session_context WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if not row:
            return {
                "session_id": session_id,
                "loaded_skills": [],
                "context_summary": "",
                "message_count": 0,
                "recent_messages": [],
            }
        import json as _json
        return {
            "session_id": row["session_id"],
            "loaded_skills": _json.loads(row["loaded_skills"]),
            "context_summary": row["context_summary"],
            "message_count": row["message_count"],
            "recent_messages": _json.loads(row["recent_messages"]),
        }

    def save_session_context(self, session_id: str, loaded_skills: list[str],
                             context_summary: str, message_count: int,
                             recent_messages: list[str] | None = None) -> None:
        """Upsert the session context after dynamic evaluation."""
        import json as _json
        msgs_json = _json.dumps(recent_messages or [])
        self._conn.execute("""
            INSERT INTO session_context
                (session_id, loaded_skills, context_summary, message_count,
                 recent_messages, updated_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(session_id) DO UPDATE SET
                loaded_skills = excluded.loaded_skills,
                context_summary = excluded.context_summary,
                message_count = excluded.message_count,
                recent_messages = excluded.recent_messages,
                updated_at = excluded.updated_at
        """, (session_id, _json.dumps(loaded_skills), context_summary,
              message_count, msgs_json))
        self._conn.commit()

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

    # ------------------------------------------------------------------
    # Implicit feedback support

    def get_session_tools(self, session_id: str) -> list[str]:
        """Return distinct tool_used values logged for a session."""
        rows = self._conn.execute(
            "SELECT DISTINCT tool_used FROM session_log WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        return [r["tool_used"] for r in rows]

    def record_implicit_feedback(self, session_id: str,
                                 loaded_skills: list[str],
                                 tools_used: list[str]) -> dict:
        """Infer feedback from skill load vs tool usage overlap and record it.

        Heuristic:
        - If the skill domain keyword appears in any tool_used string → positive
        - Skills loaded but with zero domain overlap and session had real tool
          calls (not just trivial ones) → negative

        Returns {"positive": [...], "negative": [...], "skipped": [...]}
        """
        if not loaded_skills or not tools_used:
            return {"positive": [], "negative": [], "skipped": []}

        tools_blob = " ".join(tools_used).lower()
        positive: list[str] = []
        negative: list[str] = []
        skipped: list[str] = []

        for skill_id in loaded_skills:
            # Extract the short name from "plugin:skill-name" or just "skill-name"
            short = skill_id.split(":")[-1].replace("-", " ").replace("_", " ").lower()
            # Domain keywords: individual words ≥ 4 chars
            keywords = [w for w in short.split() if len(w) >= 4]
            if not keywords:
                skipped.append(skill_id)
                continue

            hit = any(kw in tools_blob for kw in keywords)
            if hit:
                positive.append(skill_id)
            else:
                negative.append(skill_id)

        # Record against a synthetic "session-end" query vector
        # We use a zero vector placeholder; the EMA on feedback_score is what matters
        zero_vec: list[float] = []
        for skill_id in positive:
            self.record_feedback(skill_id, f"session:{session_id}", zero_vec, helpful=True)
        for skill_id in negative:
            self.record_feedback(skill_id, f"session:{session_id}", zero_vec, helpful=False)

        return {"positive": positive, "negative": negative, "skipped": skipped}

    # ------------------------------------------------------------------
    # Training data export

    def export_training_data(self) -> dict:
        """Export all available signal as structured training pairs.

        Returns a dict with keys:
          feedback_pairs  — (query, skill_content, label) from explicit feedback
          triage_pairs    — (message, action) from triage_log
          compact_pairs   — (summary, compact_digest) from closed tasks
        """
        # 1. Explicit feedback pairs
        feedback_pairs: list[dict] = []
        rows = self._conn.execute("""
            SELECT f.query, f.helpful, s.content, s.description, s.id
            FROM feedback f
            JOIN skills s ON s.id = f.skill_id
            WHERE f.query != '' AND s.content != ''
            ORDER BY f.created_at DESC
        """).fetchall()
        for row in rows:
            feedback_pairs.append({
                "query": row["query"],
                "skill_id": row["id"],
                "skill_description": row["description"] or "",
                "skill_content": (row["content"] or "")[:1000],
                "label": bool(row["helpful"]),
            })

        # 2. Triage decision pairs
        triage_pairs: list[dict] = []
        rows = self._conn.execute("""
            SELECT message_preview, action, confidence
            FROM triage_log
            WHERE action != '' AND message_preview != ''
            ORDER BY created_at DESC
        """).fetchall()
        for row in rows:
            triage_pairs.append({
                "message": row["message_preview"],
                "action": row["action"],
                "confidence": row["confidence"],
            })

        # 3. Compaction pairs (summary → compact digest)
        compact_pairs: list[dict] = []
        rows = self._conn.execute("""
            SELECT title, summary, compact
            FROM tasks
            WHERE status = 'closed' AND compact IS NOT NULL AND compact != ''
            ORDER BY closed_at DESC
        """).fetchall()
        for row in rows:
            compact_pairs.append({
                "title": row["title"],
                "input": row["summary"],
                "output": row["compact"],
            })

        return {
            "feedback_pairs": feedback_pairs,
            "triage_pairs": triage_pairs,
            "compact_pairs": compact_pairs,
        }

    def close(self) -> None:
        self._conn.close()
