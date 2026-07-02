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
import logging
import math
import re
import socket
import sqlite3
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path

_log = logging.getLogger(__name__)

# Common English + generic-imperative words that pollute BM25 keyword search
# (they match almost every skill's prose). Filtered out of ``search_fts`` so the
# deterministic fallback ranks on meaningful terms, not filler.
_FTS_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "all", "for", "with", "you", "your", "our", "are", "was",
    "this", "that", "then", "than", "into", "onto", "from", "have", "has",
    "not", "but", "can", "may", "will", "would", "should", "could", "any",
    "run", "use", "using", "get", "got", "let", "via", "per", "out", "off",
    "please", "also", "them", "they", "some", "such", "when", "what", "which",
    "how", "who", "why", "where", "want", "need", "make", "made", "done",
})


def _resolve_node_id() -> str:
    """Resolve this host's federation node_id.

    Order: ``federation.node_id`` config value → ``$SKILL_HUB_NODE_ID`` env var
    → ``socket.gethostname()`` → ``"local"``. Sanitized to a safe identifier so
    it can be used in attached-DB aliases and cross-host queries.

    Federation-lite (M4-3) treats this as an opaque tag: SQLite stores it on
    every row of ``events`` and ``tasks`` so that, when two databases live on
    the same disk (via Syncthing / rsync / git-annex), rows can be filtered or
    grouped by originating host without any coordination protocol.
    """
    import os

    raw: str | None = None
    try:
        from . import config as _cfg

        cfg = _cfg.load_config()
        fed = cfg.get("federation") or {}
        if isinstance(fed, dict):
            val = fed.get("node_id")
            if isinstance(val, str) and val.strip():
                raw = val.strip()
    except Exception:  # noqa: BLE001 — config is optional during early init
        pass

    if not raw:
        env = os.environ.get("SKILL_HUB_NODE_ID", "").strip()
        if env:
            raw = env
    if not raw:
        try:
            raw = socket.gethostname() or "local"
        except Exception:  # noqa: BLE001
            raw = "local"

    # Allow letters, digits, underscore, hyphen, dot. Replace anything else
    # with underscore so the value is safe inside SQL identifiers / file names.
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_")
    return cleaned or "local"

try:  # Optional: sqlite-vec native ANN extension.
    import sqlite_vec  # type: ignore
except ImportError:  # pragma: no cover
    sqlite_vec = None  # type: ignore

# Default dim (nomic-embed-text). The active dim is resolved per-store
# instance in SkillStore._vec_dim; this alias is kept for backwards compat.
DEFAULT_VEC_DIM = 768
VEC_DIM = DEFAULT_VEC_DIM  # back-compat alias — use self._vec_dim inside SkillStore

DB_PATH = Path.home() / ".claude" / "mcp-skill-hub" / "skill_hub.db"
SESSION_CONTEXT_FILE = Path.home() / ".claude" / "mcp-skill-hub" / "session-context.md"

# Phase M1 — level-aware retrieval weights.
# L0 ephemeral → L4 identity. See docs/plugin-extension-points.md.
LEVEL_WEIGHTS: dict[str, float] = {
    "L0": 0.3, "L1": 0.8, "L2": 1.0, "L3": 1.1, "L4": 1.3,
}
# Half-life defaults per level — recency decay exp(-age_days / half_life).
LEVEL_HALF_LIFE_DAYS: dict[str, float] = {
    "L0": 0.25, "L1": 7.0, "L2": 30.0, "L3": 180.0, "L4": 3650.0,
}

# Seeded index catalogue. Plugins may add more via plugin.json "indexes".
# ``skills`` is the built-in skill corpus (L3). ``session:log`` captures
# tool-call history at L1 for short-term recall.
_DEFAULT_VECTOR_INDEXES: dict[str, dict] = {
    "skills":              {"default_level": "L3", "half_life_days": 365.0},
    "user:identity":       {"default_level": "L4", "half_life_days": 3650.0},
    "user:preferences":    {"default_level": "L3", "half_life_days": 365.0},
    # NB: tasks are vectorised directly into tasks.vector + tasks_vec_bin/f32
    # (see _mirror_task_vec), not via a ``vectors`` namespace — so the former
    # "tasks:active"/"tasks:retrospective" seed entries were never populated and
    # have been dropped.
    "habits:tool-chains":  {"default_level": "L2", "half_life_days": 60.0},
    "habits:prompts":      {"default_level": "L2", "half_life_days": 60.0},
    "session:log":         {"default_level": "L1", "half_life_days": 3.0},
    "logs":                {"default_level": "L1", "half_life_days": 7.0},
    # LLM Wiki knowledge layer — curated pages barely decay; index is rebuilt
    # from markdown SoT via wiki_reindex, never promoted/pruned by promote_memory.
    "wiki":                {"default_level": "L3", "half_life_days": 365.0},
    "wiki-private":        {"default_level": "L3", "half_life_days": 365.0},
}


def _write_session_context_file(context_summary: str,
                                recent_messages: list[str],
                                tool_examples: list[dict] | None = None,
                                repo_context: str = "") -> None:
    """Write session context to a plain-text file for local LLM consumption.

    This file is the bridge between Claude's conversation and local tools:
    L3 skills and the L4 agent read it from disk at zero Claude token cost.

    Sections ordered by priority — truncation drops lower sections first.
    """
    try:
        SESSION_CONTEXT_FILE.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Session Context", ""]
        if context_summary:
            lines += ["## Summary", context_summary, ""]
        if recent_messages:
            lines += ["## Recent Messages"]
            for msg in recent_messages[-10:]:
                lines.append(f"- {msg}")
            lines.append("")
        if tool_examples:
            lines += ["## Recent Tool Calls"]
            for ex in tool_examples[-10:]:
                hint = ex.get("context_hint", "")
                hint_str = f" ({hint[:60]})" if hint else ""
                lines.append(
                    f"- {ex['tool_name']}: {ex['tool_input'][:120]}{hint_str}")
            lines.append("")
        if repo_context:
            lines += ["## Repo", repo_context, ""]
        SESSION_CONTEXT_FILE.write_text("\n".join(lines), encoding="utf-8")
    except OSError:
        pass


def read_session_context() -> str:
    """Read the session context file. Returns empty string if missing."""
    try:
        if SESSION_CONTEXT_FILE.exists():
            text = SESSION_CONTEXT_FILE.read_text(encoding="utf-8").strip()
            # Cap at 3000 chars — local LLMs handle this fine
            return text[:3000] if len(text) > 3000 else text
    except OSError:
        pass
    return ""


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


def compute_activity_state(last_activity_at: str | None, status: str) -> str:
    """Compute activity state: active|idle|open|closed.

    Module-level helper so it can be imported and unit-tested independently
    of SkillStore.  Thresholds are read from config at call time.
    """
    if status == "closed":
        return "closed"
    if not last_activity_at:
        return "open"
    import datetime
    try:
        last = datetime.datetime.fromisoformat(
            last_activity_at.replace("Z", "+00:00")
        )
        # SQLite datetime('now') stores UTC without tzinfo — normalise.
        if last.tzinfo is None:
            last = last.replace(tzinfo=datetime.timezone.utc)
        now = datetime.datetime.now(datetime.timezone.utc)
        age_seconds = (now - last).total_seconds()
    except Exception:
        return "open"
    from . import config as _cfg
    active_threshold = int(_cfg.get("task_activity_active_seconds") or 60)
    idle_threshold = int(_cfg.get("task_activity_idle_seconds") or 3600)
    if age_seconds <= active_threshold:
        return "active"
    if age_seconds <= idle_threshold:
        return "idle"
    return "open"


class SkillStore:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path: Path = db_path
        self.node_id: str = _resolve_node_id()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL is idempotent — re-running on an already-WAL DB is a no-op and
        # returns ``"wal"`` again. Federation-lite (M4-3) needs WAL so a
        # sibling Syncthing/rsync replica can be read concurrently without
        # blocking the local writer.
        result = self._conn.execute("PRAGMA journal_mode=WAL").fetchone()
        if result and result[0].lower() != "wal":
            _log.warning("WAL mode unavailable (filesystem may not support it); using %s", result[0])
        # Load sqlite-vec extension if available; falls back to legacy path.
        self._vec_engine: str = "legacy"
        if sqlite_vec is not None:
            try:
                self._conn.enable_load_extension(True)
                sqlite_vec.load(self._conn)
                self._conn.enable_load_extension(False)
                self._vec_engine = "sqlite-vec"
            except Exception as exc:  # noqa: BLE001
                _log.warning("sqlite-vec load failed, using legacy search: %s", exc)
        # Legacy in-process vector cache (still used when vec engine unavailable).
        self._vec_cache: dict[str, tuple[list[float], float]] = {}
        self._vec_cache_valid: bool = False
        # Active vector dimension — resolved per-store in _migrate (issue #35).
        # None until the first write or until existing data is inspected.
        self._vec_dim: int | None = None
        self._vec_dim_warned: bool = False
        self._migrate()
        if self._vec_engine == "sqlite-vec":
            try:
                self._backfill_vec_tables()
            except Exception as exc:  # noqa: BLE001
                _log.warning("vec0 backfill failed: %s", exc)
                self._vec_engine = "legacy"

    # ------------------------------------------------------------------
    # Meta table helpers (issue #35)
    # ------------------------------------------------------------------

    def _meta_get(self, key: str) -> str | None:
        """Read a value from the meta table; returns None if missing or table absent."""
        try:
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            ).fetchone()
            return row[0] if row else None
        except Exception:  # noqa: BLE001
            return None

    def _meta_set(self, key: str, value: str) -> None:
        """Write a key/value pair to the meta table."""
        self._conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._conn.commit()

    def _detect_embedding_dim(self) -> int | None:
        """Infer the embedding dim from existing stored vectors without loading a model."""
        try:
            row = self._conn.execute(
                "SELECT vector FROM embeddings WHERE vector IS NOT NULL LIMIT 1"
            ).fetchone()
            if row:
                return len(json.loads(row[0]))
        except Exception:  # noqa: BLE001
            pass
        try:
            row = self._conn.execute(
                "SELECT vector FROM tasks WHERE vector IS NOT NULL LIMIT 1"
            ).fetchone()
            if row:
                return len(json.loads(row[0]))
        except Exception:  # noqa: BLE001
            pass
        return None

    def _vec0_declared_dim(self, table: str) -> int | None:
        """Return the dimension declared in the CREATE VIRTUAL TABLE … vec0(…) DDL."""
        try:
            row = self._conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if row and row[0]:
                m = re.search(r"(?:bit|float)\[(\d+)\]", row[0])
                if m:
                    return int(m.group(1))
        except Exception:  # noqa: BLE001
            pass
        return None

    def _ensure_vec_tables(self, dim: int) -> None:
        """Create the six vec0 virtual tables at ``dim`` if they do not exist yet.

        Sets ``self._vec_dim`` on first call. Skips (with a one-time warning) when
        ``dim`` differs from the already-established dim — the mismatch was already
        logged by the write path before this call.
        """
        if self._vec_engine != "sqlite-vec":
            return
        if self._vec_dim is None:
            self._vec_dim = dim
            self._meta_set("vec_dim", str(dim))
        if dim != self._vec_dim:
            return
        d = self._vec_dim
        # S1.1 + S6 F-MEM — skills / tasks / teachings (binary KNN + float32 rerank)
        self._conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS skills_vec_bin USING vec0(
                skill_id TEXT PRIMARY KEY,
                embedding bit[{d}]
            )
        """)
        self._conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS skills_vec_f32 USING vec0(
                skill_id TEXT PRIMARY KEY,
                embedding float[{d}]
            )
        """)
        self._conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS tasks_vec_bin USING vec0(
                task_id INTEGER PRIMARY KEY,
                embedding bit[{d}]
            )
        """)
        self._conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS tasks_vec_f32 USING vec0(
                task_id INTEGER PRIMARY KEY,
                embedding float[{d}]
            )
        """)
        self._conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS teachings_vec_bin USING vec0(
                teaching_id INTEGER PRIMARY KEY,
                embedding bit[{d}]
            )
        """)
        self._conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS teachings_vec_f32 USING vec0(
                teaching_id INTEGER PRIMARY KEY,
                embedding float[{d}]
            )
        """)
        self._conn.commit()

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

            -- Per-skill injection log: one row each time search_skills returns
            -- a skill to the model. Used to compute the dashboard "inj" column
            -- (actual usage, independent of feedback).
            CREATE TABLE IF NOT EXISTS skill_injections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                skill_id    TEXT NOT NULL,
                query       TEXT,
                session_id  TEXT,
                created_at  TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_skill_inj_skill
                ON skill_injections (skill_id);
            CREATE INDEX IF NOT EXISTS idx_skill_inj_created
                ON skill_injections (created_at);

            -- Semantic response cache: store Claude Q→A pairs for re-use
            CREATE TABLE IF NOT EXISTS response_cache (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                query           TEXT NOT NULL,
                query_vector    TEXT NOT NULL,   -- JSON float array
                response        TEXT NOT NULL,
                session_id      TEXT,
                hit_count       INTEGER NOT NULL DEFAULT 0,
                quality         REAL NOT NULL DEFAULT 1.0,  -- LLM-verified freshness
                created_at      TEXT DEFAULT (datetime('now')),
                last_used_at    TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_response_cache_session
                ON response_cache (session_id);

            -- Error pattern cache: past errors → fixes
            CREATE TABLE IF NOT EXISTS error_cache (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                error_text      TEXT NOT NULL,
                error_vector    TEXT NOT NULL,   -- JSON float array
                fix_hint        TEXT NOT NULL,
                session_id      TEXT,
                confirmed       INTEGER NOT NULL DEFAULT 0,  -- 1 = user confirmed fix worked
                hit_count       INTEGER NOT NULL DEFAULT 0,
                created_at      TEXT DEFAULT (datetime('now'))
            );

            -- Prompt patterns: successful prompt structures learned per project/context
            CREATE TABLE IF NOT EXISTS prompt_patterns (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                trigger_text    TEXT NOT NULL,   -- what the user wrote
                trigger_vector  TEXT NOT NULL,
                pattern_text    TEXT NOT NULL,   -- the enriched form that worked
                context_type    TEXT,            -- "refactor", "debug", "explain", etc.
                success_count   INTEGER NOT NULL DEFAULT 1,
                created_at      TEXT DEFAULT (datetime('now'))
            );

            -- Recurring message patterns: track repeat queries for auto-skill generation
            CREATE TABLE IF NOT EXISTS message_patterns (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical       TEXT NOT NULL,   -- representative form
                vector          TEXT NOT NULL,
                count           INTEGER NOT NULL DEFAULT 1,
                skill_generated INTEGER NOT NULL DEFAULT 0,  -- 1 = auto-skill created
                last_seen_at    TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_message_patterns_count
                ON message_patterns (count DESC);

            -- Context Bridge: captured tool calls from Claude (or any AI)
            CREATE TABLE IF NOT EXISTS tool_examples (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL,
                tool_name       TEXT NOT NULL,       -- "Bash", "Read", "Grep", etc.
                tool_input      TEXT NOT NULL,        -- JSON of input (truncated)
                output_summary  TEXT,                 -- first 200 chars of result
                context_hint    TEXT,                 -- what the user was working on
                repo_path       TEXT,                 -- working directory / repo root
                category        TEXT DEFAULT 'general',  -- git, github, search, file, shell
                created_at      TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_tool_ex_tool
                ON tool_examples (tool_name);
            CREATE INDEX IF NOT EXISTS idx_tool_ex_session
                ON tool_examples (session_id);
            CREATE INDEX IF NOT EXISTS idx_tool_ex_repo
                ON tool_examples (repo_path);
            CREATE INDEX IF NOT EXISTS idx_tool_ex_category
                ON tool_examples (category);

            -- Context Bridge: accumulated per-repo knowledge
            CREATE TABLE IF NOT EXISTS repo_context (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_path       TEXT NOT NULL UNIQUE,
                commit_style    TEXT,                 -- "conventional", "freeform", etc.
                common_commands TEXT,                 -- JSON: most-used commands
                project_summary TEXT,                 -- what this project is about
                tool_stats      TEXT,                 -- JSON: tool usage aggregates
                updated_at      TEXT DEFAULT (datetime('now'))
            );

            -- Skill Evolution: version history for shadow learning
            CREATE TABLE IF NOT EXISTS skill_versions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                skill_name      TEXT NOT NULL,
                version         INTEGER NOT NULL DEFAULT 1,
                skill_json      TEXT NOT NULL,        -- full JSON snapshot before change
                change_reason   TEXT,                 -- LLM explanation of what changed
                claude_example  TEXT,                 -- what Claude did that triggered it
                local_example   TEXT,                 -- what local LLM produced
                session_id      TEXT,
                created_at      TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_skill_ver_name
                ON skill_versions (skill_name);

            -- Plugin extension-point: A7 — per-plugin schema bootstrap tracking.
            CREATE TABLE IF NOT EXISTS plugin_migrations (
                namespace   TEXT PRIMARY KEY,
                schema_hash TEXT NOT NULL,
                applied_at  TEXT DEFAULT (datetime('now'))
            );

            -- Plugin extension-point: A8 — namespaced vector corpus.
            -- Generalises `embeddings` for non-skill documents. Existing
            -- embeddings table is unchanged (backward compat).
            CREATE TABLE IF NOT EXISTS vectors (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace   TEXT NOT NULL,
                doc_id      TEXT NOT NULL,
                model       TEXT,
                vector      TEXT NOT NULL,   -- JSON float array
                norm        REAL NOT NULL DEFAULT 0.0,
                metadata    TEXT,            -- JSON blob (optional)
                indexed_at  TEXT DEFAULT (datetime('now')),
                UNIQUE(namespace, doc_id)
            );
            CREATE INDEX IF NOT EXISTS idx_vectors_namespace
                ON vectors (namespace);

            -- Plugin extension-point: A5 — scheduled task enablement state.
            CREATE TABLE IF NOT EXISTS plugin_task_state (
                plugin      TEXT NOT NULL,
                name        TEXT NOT NULL,
                enabled     INTEGER NOT NULL DEFAULT 0,
                cron        TEXT,
                external_id TEXT,
                updated_at  TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (plugin, name)
            );

            -- Phase M1 — per-index config for multi-level vector corpus.
            -- ``name`` is the vector namespace (e.g. "skills", "career:profile",
            -- "habits:tool-chains"). ``default_level`` and ``half_life_days``
            -- drive level-aware retrieval scoring in search_vectors.
            CREATE TABLE IF NOT EXISTS vector_index_config (
                name              TEXT PRIMARY KEY,
                embedding_model   TEXT,
                chunk_size        INTEGER NOT NULL DEFAULT 0,
                chunk_overlap     INTEGER NOT NULL DEFAULT 0,
                default_level     TEXT NOT NULL DEFAULT 'L2',
                half_life_days    REAL NOT NULL DEFAULT 30.0,
                max_docs          INTEGER NOT NULL DEFAULT 0,
                summarizer_prompt TEXT,
                updated_at        TEXT DEFAULT (datetime('now'))
            );

            -- Phase M1 — append-only audit trail for memory promotions/prunes.
            CREATE TABLE IF NOT EXISTS memory_audit (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                at         TEXT DEFAULT (datetime('now')),
                action     TEXT NOT NULL,  -- promote / prune / merge / identity
                namespace  TEXT,
                doc_id     TEXT,
                from_level TEXT,
                to_level   TEXT,
                reason     TEXT
            );

            -- M2 W1 / M4-3 federation-lite — durable event log.
            -- Append-only record of tool invocations + config changes that
            -- can be replayed on wake_session (M2 W2) and joined across hosts
            -- (Federation-lite) by ``node_id``. Schema matches the design in
            -- docs/design/managed-agents-refactor.md, with ``source`` renamed
            -- to the more explicit ``node_id`` per issue #22.
            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                ts          REAL NOT NULL,
                kind        TEXT NOT NULL,
                tool_name   TEXT,
                payload     TEXT NOT NULL,
                node_id     TEXT NOT NULL DEFAULT 'local'
            );
            CREATE INDEX IF NOT EXISTS idx_events_session ON events (session_id, ts);
            CREATE INDEX IF NOT EXISTS idx_events_kind    ON events (kind, ts);
            CREATE INDEX IF NOT EXISTS idx_events_node    ON events (node_id);

            -- S3 F-SELECT: profile-based plugin curation
            CREATE TABLE IF NOT EXISTS profiles (
                name         TEXT PRIMARY KEY,
                plugins_json TEXT NOT NULL,         -- JSON array of enabledPlugins entries
                description  TEXT,
                is_active    INTEGER NOT NULL DEFAULT 0,
                created_at   TEXT DEFAULT (datetime('now')),
                updated_at   TEXT DEFAULT (datetime('now'))
            );

            -- S4 F-ROUTE: ε-greedy bandit over model tiers
            CREATE TABLE IF NOT EXISTS model_rewards (
                task_class   TEXT NOT NULL,      -- trivial|simple|moderate|complex
                domain       TEXT NOT NULL,      -- primary domain hint or "_none"
                tier         TEXT NOT NULL,      -- tier_cheap|tier_mid|tier_smart
                trials       INTEGER NOT NULL DEFAULT 0,
                successes    REAL    NOT NULL DEFAULT 0.0,   -- EMA-compatible (partial rewards allowed)
                updated_at   TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (task_class, domain, tier)
            );

            -- Async background job queue (memory optimisation, classify, rerank)
            CREATE TABLE IF NOT EXISTS background_jobs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                kind         TEXT NOT NULL,
                payload      TEXT NOT NULL DEFAULT '{}',
                priority     INTEGER NOT NULL DEFAULT 5,
                status       TEXT NOT NULL DEFAULT 'pending'
                                 CHECK(status IN ('pending','running','done','failed','deferred')),
                worker_used  TEXT,
                result       TEXT,
                error        TEXT,
                attempts     INTEGER NOT NULL DEFAULT 0,
                created_at   TEXT NOT NULL DEFAULT (datetime('now')),
                started_at   TEXT,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_background_jobs_status_priority
                ON background_jobs(status, priority, created_at);

            -- FTS5 full-text search for tasks (BM25 fallback when embeddings unavailable)
            CREATE VIRTUAL TABLE IF NOT EXISTS tasks_fts USING fts5(
                title, summary, compact, tags,
                content='tasks', content_rowid='id',
                tokenize='unicode61 remove_diacritics 2'
            );

            -- FTS5 full-text search for teachings
            CREATE VIRTUAL TABLE IF NOT EXISTS teachings_fts USING fts5(
                rule, action,
                content='teachings', content_rowid='id',
                tokenize='unicode61 remove_diacritics 2'
            );

            -- Keep tasks_fts in sync with triggers
            CREATE TRIGGER IF NOT EXISTS tasks_fts_insert AFTER INSERT ON tasks BEGIN
                INSERT INTO tasks_fts(rowid, title, summary, compact, tags)
                VALUES (new.id, new.title, new.summary, new.compact, new.tags);
            END;
            CREATE TRIGGER IF NOT EXISTS tasks_fts_delete AFTER DELETE ON tasks BEGIN
                INSERT INTO tasks_fts(tasks_fts, rowid, title, summary, compact, tags)
                VALUES('delete', old.id, old.title, old.summary, old.compact, old.tags);
            END;
            CREATE TRIGGER IF NOT EXISTS tasks_fts_update AFTER UPDATE ON tasks BEGIN
                INSERT INTO tasks_fts(tasks_fts, rowid, title, summary, compact, tags)
                VALUES('delete', old.id, old.title, old.summary, old.compact, old.tags);
                INSERT INTO tasks_fts(rowid, title, summary, compact, tags)
                VALUES (new.id, new.title, new.summary, new.compact, new.tags);
            END;

            -- Keep teachings_fts in sync with triggers
            CREATE TRIGGER IF NOT EXISTS teachings_fts_insert AFTER INSERT ON teachings BEGIN
                INSERT INTO teachings_fts(rowid, rule, action)
                VALUES (new.id, new.rule, new.action);
            END;
            CREATE TRIGGER IF NOT EXISTS teachings_fts_delete AFTER DELETE ON teachings BEGIN
                INSERT INTO teachings_fts(teachings_fts, rowid, rule, action)
                VALUES('delete', old.id, old.rule, old.action);
            END;
            CREATE TRIGGER IF NOT EXISTS teachings_fts_update AFTER UPDATE ON teachings BEGIN
                INSERT INTO teachings_fts(teachings_fts, rowid, rule, action)
                VALUES('delete', old.id, old.rule, old.action);
                INSERT INTO teachings_fts(rowid, rule, action)
                VALUES (new.id, new.rule, new.action);
            END;

            -- FTS5 full-text search for skills (BM25 fallback when embeddings unavailable).
            -- ``skills.id`` is TEXT, so we use a contentless FTS table (no content/content_rowid
            -- linkage) and manage sync via triggers that delete-by-id then re-insert.
            CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(
                skill_id UNINDEXED, name, description, content,
                tokenize='unicode61 remove_diacritics 2'
            );

            CREATE TRIGGER IF NOT EXISTS skills_fts_insert AFTER INSERT ON skills BEGIN
                INSERT INTO skills_fts(skill_id, name, description, content)
                VALUES (new.id, new.name, new.description, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS skills_fts_delete AFTER DELETE ON skills BEGIN
                DELETE FROM skills_fts WHERE skill_id = old.id;
            END;
            CREATE TRIGGER IF NOT EXISTS skills_fts_update AFTER UPDATE ON skills BEGIN
                DELETE FROM skills_fts WHERE skill_id = old.id;
                INSERT INTO skills_fts(skill_id, name, description, content)
                VALUES (new.id, new.name, new.description, new.content);
            END;

            -- FTS5 full-text search for plugins (BM25 fallback when embeddings unavailable).
            CREATE VIRTUAL TABLE IF NOT EXISTS plugins_fts USING fts5(
                plugin_id UNINDEXED, short_name, description,
                tokenize='unicode61 remove_diacritics 2'
            );

            CREATE TRIGGER IF NOT EXISTS plugins_fts_insert AFTER INSERT ON plugins BEGIN
                INSERT INTO plugins_fts(plugin_id, short_name, description)
                VALUES (new.id, new.short_name, new.description);
            END;
            CREATE TRIGGER IF NOT EXISTS plugins_fts_delete AFTER DELETE ON plugins BEGIN
                DELETE FROM plugins_fts WHERE plugin_id = old.id;
            END;
            CREATE TRIGGER IF NOT EXISTS plugins_fts_update AFTER UPDATE ON plugins BEGIN
                DELETE FROM plugins_fts WHERE plugin_id = old.id;
                INSERT INTO plugins_fts(plugin_id, short_name, description)
                VALUES (new.id, new.short_name, new.description);
            END;

            -- Cron scheduler: job definitions with execution history.
            CREATE TABLE IF NOT EXISTS cron_jobs (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                name                TEXT NOT NULL UNIQUE,
                schedule            TEXT NOT NULL,
                command             TEXT NOT NULL,
                enabled             INTEGER NOT NULL DEFAULT 1,
                last_run_at         TEXT,
                last_status         TEXT,
                last_error          TEXT,
                last_duration_ms    INTEGER,
                run_count           INTEGER NOT NULL DEFAULT 0,
                created_at          TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- Pre-conversation 4-tier pipeline telemetry.
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id       TEXT,
                task_id          INTEGER,
                tier1_ms         INTEGER,
                tier2_ms         INTEGER,
                tier3_ms         INTEGER,
                tier4_ms         INTEGER,
                fallbacks_used   TEXT,           -- JSON list of tier names that fell back
                top_similarity   REAL,           -- L2: best task similarity score
                token_cost_usd   REAL,
                created_at       TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- Phase B.10: Experimentation framework — named pipeline presets.
            CREATE TABLE IF NOT EXISTS pipeline_presets (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL UNIQUE,
                description TEXT,
                config_json TEXT NOT NULL DEFAULT '{}',
                is_builtin  INTEGER DEFAULT 0,
                created_at  TEXT DEFAULT (datetime('now')),
                updated_at  TEXT DEFAULT (datetime('now'))
            );

            -- Phase B.10: A/B experiment definitions.
            CREATE TABLE IF NOT EXISTS experiments (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                name           TEXT NOT NULL,
                preset_a_id    INTEGER,
                preset_b_id    INTEGER,
                target_runs    INTEGER DEFAULT 10,
                completed_runs INTEGER DEFAULT 0,
                status         TEXT DEFAULT 'active',
                notes          TEXT,
                started_at     TEXT DEFAULT (datetime('now')),
                ended_at       TEXT,
                FOREIGN KEY (preset_a_id) REFERENCES pipeline_presets(id),
                FOREIGN KEY (preset_b_id) REFERENCES pipeline_presets(id)
            );

            -- Phase B.10: Individual runs belonging to an experiment.
            CREATE TABLE IF NOT EXISTS experiment_runs (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id       INTEGER NOT NULL,
                session_id          TEXT,
                preset_tag          TEXT CHECK(preset_tag IN ('A','B')),
                tier_durations_json TEXT,
                token_cost_usd      REAL,
                user_rating         INTEGER,
                ran_at              TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            );

            -- LLM Wiki knowledge layer — derived from markdown pages (SoT).
            -- Fully rebuildable by wiki_reindex; never hand-edited.
            CREATE TABLE IF NOT EXISTS wiki_pages (
                slug        TEXT PRIMARY KEY,        -- globally unique
                id          TEXT NOT NULL,           -- ULID, stable across rename
                title       TEXT NOT NULL,
                type        TEXT NOT NULL,
                scope       TEXT NOT NULL DEFAULT 'public',  -- public | private
                projects    TEXT NOT NULL,           -- JSON array
                tags        TEXT, aliases TEXT,       -- JSON arrays
                rel_path    TEXT NOT NULL,           -- relative to wiki_root
                updated     TEXT,
                indexed_at  TEXT DEFAULT (datetime('now'))
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_wiki_pages_id ON wiki_pages (id);

            CREATE TABLE IF NOT EXISTS wiki_edges (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                src_slug  TEXT NOT NULL,
                dst_slug  TEXT NOT NULL,             -- post-alias-resolution
                dst_raw   TEXT NOT NULL,             -- exactly what was inside [[ ]]
                edge_kind TEXT NOT NULL DEFAULT 'wikilink',  -- wikilink | alias | embed
                project   TEXT,                      -- src page's projects[0]
                resolved  INTEGER NOT NULL DEFAULT 1,-- 0 = dangling (dst not in wiki_pages)
                UNIQUE(src_slug, dst_slug, edge_kind)
            );
            CREATE INDEX IF NOT EXISTS idx_wiki_edges_dst ON wiki_edges (dst_slug);
            CREATE INDEX IF NOT EXISTS idx_wiki_edges_src ON wiki_edges (src_slug);
            CREATE INDEX IF NOT EXISTS idx_wiki_edges_resolved ON wiki_edges (resolved);

            -- Wave 2: approval queue for automatic ingest source-selection.
            -- The scanner proposes candidates (pending); the operator approves;
            -- only then does the LLM ingest spend tokens. Auto-select, not auto-spend.
            CREATE TABLE IF NOT EXISTS wiki_queue (
                slug         TEXT PRIMARY KEY,          -- source page to (re)distill
                title        TEXT NOT NULL DEFAULT '',
                scope        TEXT NOT NULL DEFAULT 'public',
                reason       TEXT NOT NULL DEFAULT '',  -- undistilled | stale
                est_calls    INTEGER NOT NULL DEFAULT 1,
                status       TEXT NOT NULL DEFAULT 'pending',  -- pending|approved|done|skipped
                diff_preview TEXT,                       -- last dry-run diff JSON
                created_at   TEXT DEFAULT (datetime('now')),
                decided_at   TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_wiki_queue_status ON wiki_queue (status);
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
        if "transcript_offset" not in ctx_cols:
            self._conn.execute(
                "ALTER TABLE session_context ADD COLUMN "
                "transcript_offset INTEGER NOT NULL DEFAULT 0"
            )
            self._conn.commit()

        # Issue #127 — session<->task link was one-way (tasks.session_id only);
        # a resumed session had no way to look up its task. Additive column +
        # one-time backfill from the existing tasks.session_id match.
        if "task_id" not in ctx_cols:
            self._conn.execute(
                "ALTER TABLE session_context ADD COLUMN task_id INTEGER"
            )
            self._conn.commit()
            self._conn.execute(
                "UPDATE session_context SET task_id = ("
                "  SELECT id FROM tasks"
                "  WHERE tasks.session_id = session_context.session_id"
                "  AND tasks.status = 'open'"
                "  ORDER BY tasks.created_at DESC LIMIT 1"
                ") WHERE task_id IS NULL"
            )
            self._conn.commit()

        # Per-task auto-approve toggle (permissive override).
        task_cols = {row[1] for row in self._conn.execute("PRAGMA table_info(tasks)")}
        if "auto_approve" not in task_cols:
            self._conn.execute(
                "ALTER TABLE tasks ADD COLUMN auto_approve INTEGER"
            )
            self._conn.commit()
            task_cols = task_cols | {"auto_approve"}

        # Per-task options blob: JSON object with routing_disabled, model_pin, etc.
        if "options" not in task_cols:
            self._conn.execute(
                "ALTER TABLE tasks ADD COLUMN options TEXT"
            )
            self._conn.commit()
            task_cols = task_cols | {"options"}

        # Phase B.9 — heartbeat: tracks last activity timestamp per open task.
        if "last_activity_at" not in task_cols:
            try:
                self._conn.execute(
                    "ALTER TABLE tasks ADD COLUMN last_activity_at TEXT"
                )
                self._conn.commit()
            except Exception:
                pass  # column already exists
        # Index on last_activity_at for open tasks (idempotent).
        try:
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_last_activity "
                "ON tasks(last_activity_at DESC) WHERE status='open'"
            )
            self._conn.commit()
        except Exception:
            pass  # partial index unsupported on some SQLite builds

        # Session-bind: cwd + branch captured on task creation for resume matching.
        if "cwd" not in task_cols:
            try:
                self._conn.execute("ALTER TABLE tasks ADD COLUMN cwd TEXT")
                self._conn.commit()
                task_cols = task_cols | {"cwd"}
            except Exception:
                pass
        if "branch" not in task_cols:
            try:
                self._conn.execute("ALTER TABLE tasks ADD COLUMN branch TEXT")
                self._conn.commit()
                task_cols = task_cols | {"branch"}
            except Exception:
                pass
        try:
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_cwd_branch_open "
                "ON tasks(cwd, branch) WHERE status = 'open'"
            )
            self._conn.commit()
        except Exception:
            pass

        # Worktree spec (JSON blob) — populated when a task owns a git worktree.
        if "worktree" not in task_cols:
            try:
                self._conn.execute("ALTER TABLE tasks ADD COLUMN worktree TEXT")
                self._conn.commit()
                task_cols = task_cols | {"worktree"}
            except Exception:
                pass

        # Status colour: short label (`green`, `yellow`, `red`, `cyan`, `blue`,
        # `gray`) used by the dashboard + listings to convey state at a glance.
        # Auto-derived for memory-index-created tasks; can be overridden via
        # ``update_task(color=...)``.
        if "color" not in task_cols:
            try:
                self._conn.execute("ALTER TABLE tasks ADD COLUMN color TEXT")
                self._conn.commit()
                task_cols = task_cols | {"color"}
            except Exception:
                pass

        # M4-3 federation-lite — tag every task with the node that authored it.
        # Cross-host queries (via ``federation_view``) filter on this column to
        # answer "which tasks belong to me vs. the synced replica?" without a
        # protocol — pure schema convention over a shared/synced file system.
        if "node_id" not in task_cols:
            try:
                self._conn.execute(
                    "ALTER TABLE tasks ADD COLUMN node_id TEXT NOT NULL "
                    f"DEFAULT '{self.node_id}'"
                )
                self._conn.commit()
                task_cols = task_cols | {"node_id"}
            except Exception:
                pass
        try:
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_node ON tasks(node_id)"
            )
            self._conn.commit()
        except Exception:
            pass

        # M3 — cross-project task federation: short repo/project name tagged
        # on every task so callers can answer "what tasks are open for repo X?"
        # without manual cwd/branch grepping. Auto-captured at save_task time
        # from the worktree spec or detect_project_from_cwd; can be overridden
        # by an explicit ``repo=`` kwarg.
        if "repo" not in task_cols:
            try:
                self._conn.execute("ALTER TABLE tasks ADD COLUMN repo TEXT")
                self._conn.commit()
                task_cols = task_cols | {"repo"}
            except Exception:
                pass
        try:
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_repo ON tasks(repo)"
            )
            self._conn.commit()
        except Exception:
            pass

        # M1 — claims layer: lets multiple Claude Code sessions / swarm
        # subprocesses coordinate ownership of a task without an LLM.
        # ``claimed_by`` holds the current owner's agent_id (NULL = free);
        # ``claim_token`` is an opaque per-claim ID used to invalidate stale
        # release calls; ``claimed_at`` is the moment of the most recent
        # claim/handoff/steal; ``stealable_at`` is the wall-clock time after
        # which steal_task() may transfer ownership without consent.
        for col in ("claimed_by", "claim_token", "claimed_at", "stealable_at"):
            if col not in task_cols:
                try:
                    self._conn.execute(
                        f"ALTER TABLE tasks ADD COLUMN {col} TEXT"
                    )
                    self._conn.commit()
                    task_cols = task_cols | {col}
                except Exception:
                    pass
        try:
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_claimed_by "
                "ON tasks(claimed_by)"
            )
            self._conn.commit()
        except Exception:
            pass

        # meta table: simple key/value store for persistent store-level config.
        # Created before the vec0 block so vec_dim can be persisted across restarts.
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS meta "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._conn.commit()

        skill_cols = {row[1] for row in self._conn.execute("PRAGMA table_info(skills)")}
        if "content_hash" not in skill_cols:
            # S1.3 — enables incremental reindex (skip rows whose file content
            # + metadata hash is unchanged).
            self._conn.execute(
                "ALTER TABLE skills ADD COLUMN content_hash TEXT"
            )
            self._conn.commit()

        # M5/#35 — resolve the active vector dim WITHOUT loading a model (see #33).
        # Priority: detected from existing embedding rows > persisted in meta >
        # defer entirely for a fresh/empty DB (first write sets the dim lazily).
        #
        # Note: expected_embedding_dim() is NOT used here — applying it eagerly
        # on a fresh/empty DB would lock tables at the config-default dim before
        # any data has been written, rejecting vectors from a different model.
        # Instead, _ensure_vec_tables(len(vector)) is called on first write and
        # both creates the tables and sets self._vec_dim atomically.
        detected = self._detect_embedding_dim()
        persisted = self._meta_get("vec_dim")
        resolved: int | None = (
            detected if detected is not None
            else (int(persisted) if persisted else None)
        )

        # S1.1 — sqlite-vec virtual tables for ANN. Binary is primary (fast
        # Hamming KNN), float32 is the rerank oracle. Only created when the
        # extension loaded successfully; everything else stays legacy.
        if self._vec_engine == "sqlite-vec" and resolved is not None:
            # Rebuild if existing vec0 tables were built at a different dim.
            cur_dim = self._vec0_declared_dim("skills_vec_f32")
            if cur_dim is not None and cur_dim != resolved:
                _log.warning(
                    "embedding dim changed %s→%s — rebuilding vec0 indexes",
                    cur_dim, resolved,
                )
                for _t in (
                    "skills_vec_bin", "skills_vec_f32",
                    "tasks_vec_bin", "tasks_vec_f32",
                    "teachings_vec_bin", "teachings_vec_f32",
                ):
                    try:
                        self._conn.execute(f"DROP TABLE IF EXISTS {_t}")
                    except Exception as exc:  # noqa: BLE001
                        _log.warning("drop %s failed: %s", _t, exc)
                self._conn.commit()
            self._vec_dim = resolved
            self._meta_set("vec_dim", str(resolved))
            try:
                self._ensure_vec_tables(resolved)
            except Exception as exc:  # noqa: BLE001
                _log.warning("vec0 table creation failed: %s", exc)
                self._vec_engine = "legacy"
        # If resolved is None (empty DB, unknown model): defer — vec0 tables are
        # created lazily on first write via _ensure_vec_tables(len(vector)).

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

        # Phase M1 — vectors: add level/recency/provenance columns.
        vec_cols = {row[1] for row in self._conn.execute("PRAGMA table_info(vectors)")}
        _vec_additions = [
            ("level",         "TEXT NOT NULL DEFAULT 'L2'"),
            ("last_accessed", "TEXT"),
            ("access_count",  "INTEGER NOT NULL DEFAULT 0"),
            ("source",        "TEXT"),
            ("project",       "TEXT"),
            ("tags",          "TEXT"),  # JSON array
        ]
        for col, ddl in _vec_additions:
            if col not in vec_cols:
                self._conn.execute(f"ALTER TABLE vectors ADD COLUMN {col} {ddl}")
        # Back-fill: skills namespace → L3, everything else stays default L2.
        self._conn.execute(
            "UPDATE vectors SET level = 'L3' WHERE namespace = 'skills' AND level = 'L2'"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_vectors_level ON vectors (level)"
        )
        self._conn.commit()

        # Seed default vector_index_config rows (idempotent — INSERT OR IGNORE).
        for name, cfg in _DEFAULT_VECTOR_INDEXES.items():
            self._conn.execute(
                """
                INSERT OR IGNORE INTO vector_index_config
                    (name, default_level, half_life_days, chunk_size, chunk_overlap, max_docs)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (name, cfg["default_level"], cfg["half_life_days"],
                 cfg.get("chunk_size", 0), cfg.get("chunk_overlap", 0),
                 cfg.get("max_docs", 0)),
            )
        self._conn.commit()

        # Incremental migrations for cron_jobs table (new columns added in H.1)
        cron_cols = {row[1] for row in self._conn.execute("PRAGMA table_info(cron_jobs)")}
        _cron_additions = [
            ("description",      "TEXT"),
            ("params",           "TEXT NOT NULL DEFAULT '{}'"),
            ("is_builtin",       "INTEGER NOT NULL DEFAULT 0"),
            ("is_dangerous",     "INTEGER NOT NULL DEFAULT 0"),
            ("updated_at",       "TEXT NOT NULL DEFAULT (datetime('now'))"),
        ]
        for col, ddl in _cron_additions:
            if col not in cron_cols:
                self._conn.execute(f"ALTER TABLE cron_jobs ADD COLUMN {col} {ddl}")
        self._conn.commit()

        # Seed built-in cron jobs (idempotent — ON CONFLICT DO UPDATE).
        self._seed_builtin_cron_jobs()

        # Phase B.10 — seed built-in pipeline presets (idempotent).
        self._seed_builtin_presets()

        # session_log: SubagentStart/SubagentStop columns (Claude Code 1.0.41+).
        sl_cols = {row[1] for row in self._conn.execute(
            "PRAGMA table_info(session_log)")}
        if "agent_id" not in sl_cols:
            self._conn.execute("ALTER TABLE session_log ADD COLUMN agent_id TEXT")
        if "agent_type" not in sl_cols:
            self._conn.execute("ALTER TABLE session_log ADD COLUMN agent_type TEXT")
        if "event" not in sl_cols:
            self._conn.execute("ALTER TABLE session_log ADD COLUMN event TEXT")
        if "transcript_path" not in sl_cols:
            self._conn.execute(
                "ALTER TABLE session_log ADD COLUMN transcript_path TEXT")
        self._conn.commit()
        try:
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_log_agent "
                "ON session_log(agent_id) WHERE agent_id IS NOT NULL"
            )
            self._conn.commit()
        except Exception:
            pass

        # Phase B.11 — memory-export history log.
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS export_history (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                kind          TEXT NOT NULL CHECK(kind IN ('export','import')),
                path          TEXT,
                tables_json   TEXT,
                row_count     INTEGER,
                size_bytes    INTEGER,
                conflict_mode TEXT,
                status        TEXT DEFAULT 'completed',
                notes         TEXT,
                created_at    TEXT DEFAULT (datetime('now'))
            )
        """)
        self._conn.commit()

        # Issue #37 — typed task↔issue links + bidirectional sync table.
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS task_issue_links (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id         INTEGER NOT NULL,
                repo            TEXT NOT NULL DEFAULT '',
                issue_number    INTEGER NOT NULL,
                url             TEXT,
                state           TEXT,
                last_synced_at  TEXT,
                writeback_done  INTEGER NOT NULL DEFAULT 0,
                UNIQUE(task_id, repo, issue_number)
            );
            CREATE INDEX IF NOT EXISTS idx_til_task
                ON task_issue_links(task_id);
            CREATE INDEX IF NOT EXISTS idx_til_issue
                ON task_issue_links(repo, issue_number);
        """)
        self._conn.commit()

        # One-time idempotent tag migration: for every task with tag `issue:<n>`,
        # parse the number and create a typed link row.  INSERT OR IGNORE ensures
        # re-running is safe.
        task_col_names = {r[1] for r in self._conn.execute("PRAGMA table_info(tasks)")}
        _repo_sel = "repo" if "repo" in task_col_names else "''"
        rows = self._conn.execute(
            f"SELECT id, tags, {_repo_sel} AS repo FROM tasks WHERE tags LIKE '%issue:%'"
        ).fetchall()
        for row in rows:
            tags_str = row["tags"] or ""
            task_repo = row["repo"] or ""
            for part in tags_str.split():
                if part.startswith("issue:"):
                    # Tag is `issue:<id>` where <id> may carry a source prefix
                    # (e.g. `issue:gh:123`, `issue:text:0001`) or be a bare
                    # number (`issue:123`). Take the trailing numeric segment.
                    try:
                        num = int(part[6:].rsplit(":", 1)[-1])
                    except ValueError:
                        continue
                    self._conn.execute(
                        "INSERT OR IGNORE INTO task_issue_links "
                        "(task_id, repo, issue_number) VALUES (?, ?, ?)",
                        (row["id"], task_repo, num),
                    )
        self._conn.commit()

        # Issue #38 — Claude Code task projection: stable-key dedup + source tracking.
        # claude_task_key: stable hash or cid:<id> string (see claude_tasks.stable_key).
        # claude_task_id: the task_id from the Claude tool response (may be None).
        task_cols = {row[1] for row in self._conn.execute("PRAGMA table_info(tasks)")}
        if "claude_task_key" not in task_cols:
            try:
                self._conn.execute(
                    "ALTER TABLE tasks ADD COLUMN claude_task_key TEXT"
                )
                self._conn.commit()
                task_cols = task_cols | {"claude_task_key"}
            except Exception:
                pass
        if "claude_task_id" not in task_cols:
            try:
                self._conn.execute(
                    "ALTER TABLE tasks ADD COLUMN claude_task_id TEXT"
                )
                self._conn.commit()
            except Exception:
                pass
        try:
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_claude_key "
                "ON tasks(claude_task_key) WHERE claude_task_key IS NOT NULL"
            )
            self._conn.commit()
        except Exception:
            pass

        # Rebuild FTS5 indexes from existing data (idempotent — safe to call repeatedly).
        self._rebuild_fts_index(self._conn)

    def _rebuild_fts_index(self, conn: sqlite3.Connection) -> None:
        """Repair FTS5 indexes ONLY when they have drifted from their sources.

        All four FTS tables (skills_fts/plugins_fts/tasks_fts/teachings_fts) are
        kept current incrementally by the AFTER INSERT/UPDATE/DELETE triggers
        created in ``_migrate``. The expensive part — wiping ``skills_fts`` and
        re-tokenising every skill's ``content`` column — previously ran on EVERY
        connection. Because ``SkillStore`` is constructed at module import
        (``server.py``: ``_store = SkillStore()``), that synchronous re-index
        blocked the MCP stdio ``initialize`` handshake on large catalogs (the
        client gives up after ~30s) AND rewrote the whole FTS index into the WAL
        on every open across the webapp/CLI/hooks, ballooning it to gigabytes.

        Now we only repopulate when a cheap row-count check shows real drift
        (e.g. rows that predate the triggers); in steady state this is a no-op.
        """
        try:
            def _count(sql: str) -> int:
                row = conn.execute(sql).fetchone()
                return int(row[0]) if row and row[0] is not None else 0

            # tasks/teachings: small tables — the idempotent FTS5 'rebuild' is cheap.
            conn.execute("INSERT INTO tasks_fts(tasks_fts) VALUES('rebuild')")
            conn.execute("INSERT INTO teachings_fts(teachings_fts) VALUES('rebuild')")

            # skills_fts / plugins_fts hold large content — repopulate only on drift.
            if _count("SELECT count(*) FROM skills_fts") != _count("SELECT count(*) FROM skills"):
                conn.execute("DELETE FROM skills_fts")
                conn.execute(
                    "INSERT INTO skills_fts(skill_id, name, description, content) "
                    "SELECT id, name, description, content FROM skills"
                )
                _log.info("skills_fts drift detected — index repopulated")
            if _count("SELECT count(*) FROM plugins_fts") != _count("SELECT count(*) FROM plugins"):
                conn.execute("DELETE FROM plugins_fts")
                conn.execute(
                    "INSERT INTO plugins_fts(plugin_id, short_name, description) "
                    "SELECT id, short_name, description FROM plugins"
                )
                _log.info("plugins_fts drift detected — index repopulated")
            conn.commit()
        except sqlite3.OperationalError as exc:
            _log.warning("FTS5 rebuild failed (FTS5 may be unavailable): %s", exc)

    def _seed_builtin_presets(self) -> None:
        """Seed built-in pipeline presets (skip if already exist)."""
        presets = [
            {
                "name": "fast-local",
                "description": "All Ollama, fastest on a warm local machine",
                "config_json": {
                    "classify_backend": "ollama_qwen",
                    "embedding_backend_priority": ["ollama", "sentence_transformers"],
                    "synthesis_backend": "ollama",
                    "rewrite_backend": "none",
                    "pipeline_tier1_timeout_ms": 300,
                    "pipeline_tier2_timeout_ms": 200,
                    "pipeline_tier3_timeout_ms": 600,
                    "pipeline_tier4_timeout_ms": 0,
                },
            },
            {
                "name": "cheap-cloud",
                "description": "Haiku + SentenceTransformers, no Ollama needed (~$0.005/conv)",
                "config_json": {
                    "classify_backend": "haiku_json_then_yake",
                    "embedding_backend_priority": ["sentence_transformers"],
                    "synthesis_backend": "haiku",
                    "rewrite_backend": "none",
                    "pipeline_tier1_timeout_ms": 500,
                    "pipeline_tier2_timeout_ms": 400,
                    "pipeline_tier3_timeout_ms": 1200,
                    "pipeline_tier4_timeout_ms": 0,
                },
            },
            {
                "name": "quality-cloud",
                "description": "Sonnet everywhere (~$0.02/conv)",
                "config_json": {
                    "classify_backend": "haiku_json",
                    "embedding_backend_priority": ["sentence_transformers"],
                    "synthesis_backend": "sonnet",
                    "rewrite_backend": "sonnet",
                    "pipeline_tier1_timeout_ms": 1000,
                    "pipeline_tier2_timeout_ms": 800,
                    "pipeline_tier3_timeout_ms": 3000,
                    "pipeline_tier4_timeout_ms": 3000,
                    "pipeline_tier4_min_complexity": "low",
                },
            },
            {
                "name": "offline-only",
                "description": "YAKE + FTS5 + SentenceTransformers, zero API cost",
                "config_json": {
                    "classify_backend": "yake_keywords",
                    "embedding_backend_priority": ["sentence_transformers"],
                    "synthesis_backend": "concat",
                    "rewrite_backend": "none",
                    "pipeline_tier1_timeout_ms": 100,
                    "pipeline_tier2_timeout_ms": 200,
                    "pipeline_tier3_timeout_ms": 0,
                    "pipeline_tier4_timeout_ms": 0,
                },
            },
            {
                "name": "balanced",
                "description": "Haiku L1+L3, Ollama L2, Sonnet L4 only when complex — recommended default",
                "config_json": {
                    "classify_backend": "haiku_json_then_yake",
                    "embedding_backend_priority": ["ollama", "sentence_transformers"],
                    "synthesis_backend": "haiku",
                    "rewrite_backend": "sonnet",
                    "pipeline_tier1_timeout_ms": 500,
                    "pipeline_tier2_timeout_ms": 400,
                    "pipeline_tier3_timeout_ms": 1200,
                    "pipeline_tier4_timeout_ms": 1500,
                    "pipeline_tier4_min_complexity": "medium",
                },
            },
        ]
        for p in presets:
            try:
                self._conn.execute(
                    "INSERT OR IGNORE INTO pipeline_presets (name, description, config_json, is_builtin) "
                    "VALUES (?, ?, ?, 1)",
                    (p["name"], p["description"], json.dumps(p["config_json"])),
                )
            except Exception:  # noqa: BLE001
                pass
        self._conn.commit()

    # ------------------------------------------------------------------
    # Experiments: preset CRUD + A/B runner

    def list_presets(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, name, description, config_json, is_builtin, created_at "
            "FROM pipeline_presets ORDER BY is_builtin DESC, name"
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["config"] = json.loads(d.pop("config_json") or "{}")
            except Exception:  # noqa: BLE001
                d["config"] = {}
            result.append(d)
        return result

    def record_memory_audit(
        self,
        *,
        action: str,
        namespace: str | None = None,
        doc_id: str | None = None,
        from_level: str | None = None,
        to_level: str | None = None,
        reason_json: dict | None = None,
    ) -> int:
        """Insert a row into memory_audit. Returns the new rowid.

        reason_json is serialised into the `reason` TEXT column.
        """
        cur = self._conn.execute(
            "INSERT INTO memory_audit (action, namespace, doc_id, from_level, to_level, reason) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (action, namespace, doc_id, from_level, to_level,
             json.dumps(reason_json) if reason_json is not None else None),
        )
        self._conn.commit()
        return cur.lastrowid or 0

    def get_preset(self, preset_id: int) -> dict | None:
        r = self._conn.execute(
            "SELECT * FROM pipeline_presets WHERE id = ?", (preset_id,)
        ).fetchone()
        if not r:
            return None
        d = dict(r)
        try:
            d["config"] = json.loads(d.pop("config_json") or "{}")
        except Exception:  # noqa: BLE001
            d["config"] = {}
        return d

    def save_preset(self, name: str, description: str, config: dict) -> int:
        cur = self._conn.execute(
            "INSERT INTO pipeline_presets (name, description, config_json) VALUES (?, ?, ?)",
            (name, description, json.dumps(config)),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def delete_preset(self, preset_id: int) -> bool:
        """Delete a non-builtin preset. Returns True if deleted."""
        cur = self._conn.execute(
            "DELETE FROM pipeline_presets WHERE id = ? AND is_builtin = 0", (preset_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def list_experiments(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT e.*, pa.name AS preset_a_name, pb.name AS preset_b_name "
            "FROM experiments e "
            "LEFT JOIN pipeline_presets pa ON e.preset_a_id = pa.id "
            "LEFT JOIN pipeline_presets pb ON e.preset_b_id = pb.id "
            "ORDER BY e.started_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def create_experiment(
        self,
        name: str,
        preset_a_id: int,
        preset_b_id: int,
        target_runs: int = 10,
        notes: str = "",
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO experiments (name, preset_a_id, preset_b_id, target_runs, notes) "
            "VALUES (?,?,?,?,?)",
            (name, preset_a_id, preset_b_id, target_runs, notes),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_experiment_stats(self, experiment_id: int) -> dict:
        """Return A/B comparison stats for an experiment."""
        rows = self._conn.execute(
            "SELECT preset_tag, tier_durations_json, token_cost_usd, user_rating "
            "FROM experiment_runs WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchall()
        stats: dict = {
            "A": {"runs": 0, "avg_cost": None, "ratings": []},
            "B": {"runs": 0, "avg_cost": None, "ratings": []},
        }
        for r in rows:
            tag = r["preset_tag"] or "A"
            if tag not in stats:
                continue
            stats[tag]["runs"] += 1
            cost = r["token_cost_usd"]
            if cost is not None:
                stats[tag].setdefault("_costs", []).append(cost)
            rating = r["user_rating"]
            if rating is not None:
                stats[tag]["ratings"].append(rating)
        for tag in ("A", "B"):
            costs = stats[tag].pop("_costs", [])
            if costs:
                stats[tag]["avg_cost"] = round(sum(costs) / len(costs), 6)
            ratings = stats[tag]["ratings"]
            if ratings:
                stats[tag]["avg_rating"] = round(sum(ratings) / len(ratings), 2)
        return stats

    def cancel_experiment(self, experiment_id: int) -> bool:
        """Mark an experiment as cancelled. Returns True if the row was updated."""
        try:
            cur = self._conn.execute(
                "UPDATE experiments SET status='cancelled', ended_at=datetime('now') "
                "WHERE id=? AND status='active'",
                (experiment_id,),
            )
            self._conn.commit()
            return cur.rowcount > 0
        except Exception:  # noqa: BLE001
            return False

    def rate_experiment_run(self, run_id: int, rating: int) -> None:
        """Set user_rating on an experiment run. Rating must be 1 or -1."""
        try:
            self._conn.execute(
                "UPDATE experiment_runs SET user_rating=? WHERE id=?",
                (rating, run_id),
            )
            self._conn.commit()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Write

    def upsert_skill(self, skill: Skill, content_hash: str | None = None) -> None:
        self._conn.execute("""
            INSERT INTO skills (id, name, description, content, file_path, plugin, target, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name         = excluded.name,
                description  = excluded.description,
                content      = excluded.content,
                file_path    = excluded.file_path,
                plugin       = excluded.plugin,
                target       = excluded.target,
                content_hash = excluded.content_hash,
                indexed_at   = datetime('now')
        """, (skill.id, skill.name, skill.description, skill.content,
              skill.file_path, skill.plugin, skill.target, content_hash))
        self._conn.commit()

    def get_content_hash(self, skill_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT content_hash FROM skills WHERE id = ?", (skill_id,)
        ).fetchone()
        return row["content_hash"] if row else None

    def delete_skill(self, skill_id: str) -> None:
        """Remove a skill and all its companion rows.

        Deleting from ``skills`` cascades to ``skills_fts`` via trigger, but the
        vector tables (``embeddings``, ``skills_vec_bin``, ``skills_vec_f32``)
        are managed manually in :meth:`upsert_embedding` and must be cleaned
        here too, or they leak orphaned vectors that keep surfacing in search.
        The vec0 tables are created lazily on first embedding write (legacy
        engine never creates them at all), so they may not exist yet — best
        effort, matching :meth:`upsert_embedding`'s own lazy-create handling.
        """
        self._conn.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
        self._conn.execute("DELETE FROM embeddings WHERE skill_id = ?", (skill_id,))
        for table in ("skills_vec_bin", "skills_vec_f32"):
            try:
                self._conn.execute(f"DELETE FROM {table} WHERE skill_id = ?", (skill_id,))
            except sqlite3.OperationalError:
                pass
        self._conn.commit()
        self._vec_cache_valid = False

    def prune_skills(self, excluded_substrings: tuple[str, ...] = ()) -> list[str]:
        """Delete indexed skills whose backing file is gone or excluded.

        A skill is pruned when its ``file_path`` no longer exists on disk
        (the plugin/skill was uninstalled) or when the path contains any of
        ``excluded_substrings`` (e.g. archive dirs, transient ``temp_git_``
        marketplace clones). ``index_all`` upserts but never deletes, so
        without this the index accumulates stale rows forever. Returns the list
        of pruned skill ids.
        """
        pruned: list[str] = []
        rows = self._conn.execute(
            "SELECT id, file_path FROM skills WHERE target = 'claude'"
        ).fetchall()
        for row in rows:
            path = row["file_path"] or ""
            excluded = any(sub in path for sub in excluded_substrings)
            missing = bool(path) and not Path(path).exists()
            if excluded or missing:
                self.delete_skill(row["id"])
                pruned.append(row["id"])
        return pruned

    def dedupe_skills_by_content_hash(self) -> list[str]:
        """Collapse byte-identical skills indexed under more than one id.

        A plugin's skills can be reachable from several on-disk sources that
        each get their own ``skill_id`` (a marketplace git checkout vs. its
        installed cache copy; sibling plugins in one marketplace that share a
        single ``skills/`` tree). ``index_all`` upserts by id, so it never
        catches these — same content, different id, extra row. Group
        ``target='claude'`` rows by ``content_hash`` and keep exactly one per
        group: prefer a row whose ``file_path`` lives under
        ``plugins/cache/`` (the installed copy actually served to Claude
        Code) over one that doesn't; break remaining ties on the smaller id
        for determinism. Returns the list of deleted skill ids.
        """
        rows = self._conn.execute(
            "SELECT id, file_path, content_hash FROM skills "
            "WHERE target = 'claude' AND content_hash IS NOT NULL"
        ).fetchall()
        groups: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            groups.setdefault(row["content_hash"], []).append(row)

        def _sort_key(row: sqlite3.Row) -> tuple[bool, str]:
            is_cache = "/plugins/cache/" in (row["file_path"] or "")
            return (not is_cache, row["id"])

        deleted: list[str] = []
        for group in groups.values():
            if len(group) < 2:
                continue
            keep, *rest = sorted(group, key=_sort_key)
            for row in rest:
                self.delete_skill(row["id"])
                deleted.append(row["id"])
        return deleted

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
        # Mirror into vec0 tables (binary + float32). Best-effort — never blocks
        # the primary write. Lazy-create vec0 tables on first write (#35).
        if self._vec_engine == "sqlite-vec":
            try:
                self._ensure_vec_tables(len(vector))
            except Exception as exc:  # noqa: BLE001
                _log.debug("vec0 ensure failed for %s: %s", skill_id, exc)
            if self._vec_dim is None or len(vector) != self._vec_dim:
                if self._vec_engine == "sqlite-vec" and self._vec_dim is not None and not self._vec_dim_warned:
                    _log.warning(
                        "embedding dim mismatch: expected %d, got %d for %s — skipping vec0 write",
                        self._vec_dim, len(vector), skill_id,
                    )
                    self._vec_dim_warned = True
            else:
                try:
                    self._write_vec_rows(skill_id, vector)
                except Exception as exc:  # noqa: BLE001
                    _log.debug("vec0 upsert failed for %s: %s", skill_id, exc)

    def _write_vec_rows(self, skill_id: str, vector: list[float]) -> None:
        """Mirror a float32 embedding into both vec0 virtual tables."""
        from .embeddings import quantize_binary
        bin_blob = quantize_binary(vector)
        dim = self._vec_dim if self._vec_dim is not None else len(vector)
        f32_blob = struct.pack(f"{dim}f", *vector)
        self._conn.execute("DELETE FROM skills_vec_bin WHERE skill_id = ?", (skill_id,))
        self._conn.execute(
            "INSERT INTO skills_vec_bin (skill_id, embedding) VALUES (?, vec_bit(?))",
            (skill_id, bin_blob),
        )
        self._conn.execute("DELETE FROM skills_vec_f32 WHERE skill_id = ?", (skill_id,))
        self._conn.execute(
            "INSERT INTO skills_vec_f32 (skill_id, embedding) VALUES (?, ?)",
            (skill_id, f32_blob),
        )
        self._conn.commit()

    def _backfill_vec_tables(self) -> None:
        """Populate vec0 tables from canonical JSON-vector columns.

        Runs on startup when the vec engine is available. Skips rows already
        present (cheap idempotent) and rows with wrong dimensionality.
        Covers: skills/embeddings, tasks, teachings (S1 + S6).
        """
        # If vec dim is not yet resolved (fresh/empty DB), the vec0 tables have
        # not been created yet — skip backfill; they will be created lazily on
        # the first write via _ensure_vec_tables(len(vector)).
        if self._vec_dim is None:
            return
        # Skills
        existing = {
            row[0] for row in self._conn.execute(
                "SELECT skill_id FROM skills_vec_bin"
            ).fetchall()
        }
        rows = self._conn.execute(
            "SELECT skill_id, vector FROM embeddings"
        ).fetchall()
        added = 0
        for row in rows:
            sid = row["skill_id"]
            if sid in existing:
                continue
            try:
                vec = json.loads(row["vector"])
            except (TypeError, ValueError):
                continue
            if self._vec_dim is None or len(vec) != self._vec_dim:
                continue
            try:
                self._write_vec_rows(sid, vec)
                added += 1
            except Exception as exc:  # noqa: BLE001
                _log.debug("backfill failed for %s: %s", sid, exc)
        if added:
            _log.info("vec0 skills backfill: %d rows", added)

        # Tasks (S6)
        self._backfill_secondary_vec(
            "tasks",
            "SELECT id, vector FROM tasks WHERE vector IS NOT NULL",
            "task_id",
            "tasks_vec_bin",
            "tasks_vec_f32",
        )
        # Teachings (S6)
        self._backfill_secondary_vec(
            "teachings",
            "SELECT id, rule_vector AS vector FROM teachings WHERE rule_vector IS NOT NULL",
            "teaching_id",
            "teachings_vec_bin",
            "teachings_vec_f32",
        )

    def _backfill_secondary_vec(
        self,
        label: str,
        select_sql: str,
        id_col: str,
        bin_tbl: str,
        f32_tbl: str,
    ) -> None:
        existing = {
            row[0] for row in self._conn.execute(
                f"SELECT {id_col} FROM {bin_tbl}"
            ).fetchall()
        }
        rows = self._conn.execute(select_sql).fetchall()
        added = 0
        for row in rows:
            rid = row["id"] if "id" in row.keys() else row[0]
            if rid in existing:
                continue
            try:
                vec = json.loads(row["vector"])
            except (TypeError, ValueError):
                continue
            if self._vec_dim is None or len(vec) != self._vec_dim:
                continue
            try:
                self._write_secondary_vec_rows(rid, vec, id_col, bin_tbl, f32_tbl)
                added += 1
            except Exception as exc:  # noqa: BLE001
                _log.debug("backfill %s failed for %s: %s", label, rid, exc)
        if added:
            _log.info("vec0 %s backfill: %d rows", label, added)

    def _write_secondary_vec_rows(
        self,
        row_id: int,
        vector: list[float],
        id_col: str,
        bin_tbl: str,
        f32_tbl: str,
    ) -> None:
        """Mirror a JSON vector into ``{bin_tbl}`` + ``{f32_tbl}`` virtual tables.

        Used by tasks and teachings (S6). Keyed by ``id_col`` (integer PK).
        """
        from .embeddings import quantize_binary
        bin_blob = quantize_binary(vector)
        dim = self._vec_dim if self._vec_dim is not None else len(vector)
        f32_blob = struct.pack(f"{dim}f", *vector)
        self._conn.execute(f"DELETE FROM {bin_tbl} WHERE {id_col} = ?", (row_id,))
        self._conn.execute(
            f"INSERT INTO {bin_tbl} ({id_col}, embedding) VALUES (?, vec_bit(?))",
            (row_id, bin_blob),
        )
        self._conn.execute(f"DELETE FROM {f32_tbl} WHERE {id_col} = ?", (row_id,))
        self._conn.execute(
            f"INSERT INTO {f32_tbl} ({id_col}, embedding) VALUES (?, ?)",
            (row_id, f32_blob),
        )
        self._conn.commit()

    def log_skill_injection(self, skill_id: str, query: str = "",
                            session_id: str | None = None,
                            domain_hints: list[str] | None = None) -> int:
        """Record that a skill's content was returned to the model.

        Dispatches the ``on_skill_activated`` plugin hook so plugins can
        capture context for feedback scoring.

        Returns the injection row id.
        """
        cursor = self._conn.execute(
            "INSERT INTO skill_injections (skill_id, query, session_id) "
            "VALUES (?, ?, ?)",
            (skill_id, query[:200] if query else "", session_id),
        )
        injection_id = cursor.lastrowid
        self._conn.commit()

        try:
            from . import plugin_hooks
            plugin_hooks.dispatch(
                "on_skill_activated",
                {
                    "skill_id": skill_id,
                    "query": query,
                    "session_id": session_id or "",
                    "domain_hints": domain_hints or [],
                    "injection_id": injection_id,
                },
            )
        except Exception:
            pass

        return injection_id

    def record_skill_used(
        self,
        skill_id: str,
        session_id: str,
        injection_id: int | None = None,
    ) -> int | None:
        """Emit a ``skill.used`` event tied to the most-recent injection.

        Resolves ``injection_id`` automatically when not supplied: queries
        ``skill_injections`` for the latest row matching ``skill_id`` +
        ``session_id``.  If no matching injection exists the event is still
        written with ``injection_id=null`` so used-without-injection is
        visible.

        Returns the new event row id (or None on failure).
        """
        resolved_id = injection_id
        if resolved_id is None and session_id:
            try:
                row = self._conn.execute(
                    "SELECT id FROM skill_injections "
                    "WHERE skill_id = ? AND session_id = ? "
                    "ORDER BY id DESC LIMIT 1",
                    (skill_id, session_id),
                ).fetchone()
                if row:
                    resolved_id = int(row["id"])
            except Exception as exc:  # noqa: BLE001
                _log.debug("record_skill_used: injection lookup failed: %s", exc)

        payload: dict = {
            "skill_id": skill_id,
            "session_id": session_id,
            "injection_id": resolved_id,
            "matched": resolved_id is not None,
        }
        return self.append_event(
            session_id,
            "skill.used",
            payload,
            tool_name="search_skills",
        )

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

        When sqlite-vec is available: Hamming-KNN on binary vectors → float32
        cosine rerank → feedback boost. Otherwise falls back to the legacy
        in-process cache path.
        """
        if self._vec_engine == "sqlite-vec" and self._vec_dim is not None and len(query_vector) == self._vec_dim:
            try:
                return self._search_vec(query_vector, top_k, similarity_threshold, target)
            except Exception as exc:  # noqa: BLE001
                _log.warning("sqlite-vec search failed, falling back: %s", exc)

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

    def _search_vec(self, query_vector: list[float], top_k: int,
                    similarity_threshold: float,
                    target: str | None) -> list[dict]:
        """sqlite-vec path: binary-KNN candidates → float32 rerank → boost."""
        from . import config as _cfg
        from .embeddings import quantize_binary

        rerank_k = max(top_k, int(_cfg.get("rerank_top_k") or 20))
        qbin = quantize_binary(query_vector)

        # Stage 1: Hamming KNN on binary vectors.
        bin_rows = self._conn.execute(
            """
            SELECT skill_id, distance
            FROM skills_vec_bin
            WHERE embedding MATCH vec_bit(?) AND k = ?
            ORDER BY distance
            """,
            (qbin, rerank_k),
        ).fetchall()
        if not bin_rows:
            return []
        candidate_ids = [r["skill_id"] for r in bin_rows]

        # Stage 2: float32 cosine rerank using pre-stored norms.
        qnorm = math.sqrt(sum(x * x for x in query_vector))
        if qnorm == 0.0:
            return []

        placeholders = ",".join("?" * len(candidate_ids))
        meta_sql = f"""
            SELECT s.id, s.name, s.description, s.content, s.plugin,
                   s.target, s.feedback_score, e.vector, e.norm
            FROM skills s
            JOIN embeddings e ON e.skill_id = s.id
            WHERE s.id IN ({placeholders})
        """
        params: list = list(candidate_ids)
        if target:
            meta_sql += " AND s.target = ?"
            params.append(target)
        rows = self._conn.execute(meta_sql, params).fetchall()

        scored: list[tuple[float, dict]] = []
        for row in rows:
            snorm = row["norm"] or 0.0
            if snorm == 0.0:
                continue
            try:
                vec = json.loads(row["vector"])
            except (TypeError, ValueError):
                continue
            dot = sum(a * b for a, b in zip(query_vector, vec))
            sim = dot / (qnorm * snorm)
            if sim < similarity_threshold:
                continue
            boost = float(row["feedback_score"] or 1.0)
            d = {k: row[k] for k in ("id", "name", "description", "content",
                                      "plugin", "target", "feedback_score")}
            scored.append((sim * boost, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def search_fts(self, query_text: str, top_k: int = 3,
                   target: str | None = None) -> list[dict]:
        """Deterministic keyword search over ``skills_fts`` (BM25) — no embeddings.

        This is the offline/local-down fallback for :meth:`search`: it needs no
        Ollama and no vector engine, so skill candidates still surface when the
        embedding backend is unavailable. Returns the same dict shape as
        ``search`` (id/name/description/content/plugin/target/feedback_score),
        ranked by BM25 and boosted by the pre-aggregated feedback score.
        """
        # Build a safe FTS5 MATCH expr: FTS5 treats punctuation as syntax, so a
        # raw prompt would raise "fts5: syntax error". Extract word tokens, drop
        # stopwords/noise (which otherwise dominate BM25 and surface irrelevant
        # skills), quote each, OR them together.
        tokens = [t for t in re.findall(r"[A-Za-z0-9_]+", query_text.lower())
                  if len(t) >= 3 and t not in _FTS_STOPWORDS][:12]
        if not tokens:
            return []
        match_expr = " OR ".join(f'"{t}"' for t in dict.fromkeys(tokens))

        sql = """
            SELECT s.id, s.name, s.description, s.content, s.plugin,
                   s.target, s.feedback_score,
                   bm25(skills_fts) AS rank
            FROM skills_fts
            JOIN skills s ON s.id = skills_fts.skill_id
            WHERE skills_fts MATCH ?
        """
        params: list = [match_expr]
        if target:
            sql += " AND s.target = ?"
            params.append(target)
        sql += " ORDER BY rank LIMIT ?"
        params.append(max(top_k, 1))
        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as exc:  # malformed MATCH — never crash the hook
            _log.warning("search_fts failed: %s", exc)
            return []

        # bm25 returns negative scores (lower = better). Rank by bm25 asc, but
        # let a strong feedback boost reorder near-ties (mirrors search()).
        scored: list[tuple[float, dict]] = []
        for row in rows:
            boost = float(row["feedback_score"] or 1.0)
            # Lower bm25 is better; divide by boost so well-liked skills rise.
            scored.append((float(row["rank"]) / boost,
                           {k: row[k] for k in ("id", "name", "description",
                                                 "content", "plugin", "target",
                                                 "feedback_score")}))
        scored.sort(key=lambda x: x[0])
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
        teaching_id = cur.lastrowid or 0
        self._mirror_teaching_vec(teaching_id, rule_vector)
        return teaching_id

    def search_teachings(self, query_vector: list[float],
                         min_sim: float = 0.6,
                         top_k: int | None = None) -> list[dict]:
        """Find teachings whose rule matches the query semantically.

        Uses the vec0 binary-KNN + float32 rerank path when available (S6);
        falls back to the in-Python cosine loop otherwise.

        ``top_k`` caps the number of (best-ranked) teachings returned; ``None``
        (default) returns all matches above ``min_sim``.
        """
        if self._vec_engine == "sqlite-vec" and self._vec_dim is not None and len(query_vector) == self._vec_dim:
            vec_results = self._search_teachings_vec(query_vector, min_sim)
            if vec_results is not None:
                return vec_results[:top_k] if top_k is not None else vec_results

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
        ranked = [r for _, r in results]
        return ranked[:top_k] if top_k is not None else ranked

    def _search_teachings_vec(self, query_vector: list[float],
                              min_sim: float) -> list[dict] | None:
        """S6 vec0 path for teachings. Returns None to signal fallback on failure."""
        from . import config as _cfg
        from .embeddings import quantize_binary

        rerank_k = max(10, int(_cfg.get("rerank_top_k") or 20))
        try:
            qbin = quantize_binary(query_vector)
            bin_rows = self._conn.execute(
                """
                SELECT teaching_id, distance
                FROM teachings_vec_bin
                WHERE embedding MATCH vec_bit(?) AND k = ?
                ORDER BY distance
                """,
                (qbin, rerank_k),
            ).fetchall()
        except Exception:
            return None
        if not bin_rows:
            return []
        ids = [r["teaching_id"] for r in bin_rows]

        qnorm = math.sqrt(sum(x * x for x in query_vector))
        if qnorm == 0.0:
            return []

        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT id, rule, rule_vector, action, target_type, target_id, weight "
            f"FROM teachings WHERE id IN ({placeholders})",
            ids,
        ).fetchall()

        scored: list[tuple[float, dict]] = []
        for row in rows:
            try:
                vec = json.loads(row["rule_vector"])
            except (TypeError, ValueError):
                continue
            snorm = math.sqrt(sum(x * x for x in vec))
            if snorm == 0.0:
                continue
            dot = sum(a * b for a, b in zip(query_vector, vec))
            sim = dot / (qnorm * snorm)
            if sim < min_sim:
                continue
            scored.append((sim * float(row["weight"]), dict(row)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored]

    def list_teachings(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT id, rule, action, target_type, target_id, weight FROM teachings "
            "ORDER BY created_at DESC"
        ).fetchall()

    def remove_teaching(self, teaching_id: int) -> bool:
        cur = self._conn.execute("DELETE FROM teachings WHERE id = ?", (teaching_id,))
        self._conn.commit()
        if self._vec_engine == "sqlite-vec":
            try:
                self._conn.execute("DELETE FROM teachings_vec_bin WHERE teaching_id = ?", (teaching_id,))
                self._conn.execute("DELETE FROM teachings_vec_f32 WHERE teaching_id = ?", (teaching_id,))
                self._conn.commit()
            except Exception:
                pass
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

        # 2. Teaching-based boost — honour configurable min_sim so users can
        # tune transparent triggering vs. precision.
        from . import config as _cfg
        teach_min = float(_cfg.get("teaching_min_similarity") or 0.5)
        teachings = self.search_teachings(query_vector, min_sim=teach_min)
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

    def log_session_subagent(self, session_id: str, agent_id: str,
                             agent_type: str, event: str,
                             transcript_path: str = "") -> None:
        """Record a SubagentStart / SubagentStop event for the session.

        Stored in ``session_log`` with ``tool_used`` set to ``"subagent"`` so
        existing aggregations (which group by ``tool_used``) still work, while
        the new ``agent_id``/``agent_type``/``event`` columns let
        ``session_stats()`` and downstream analyses break out the timeline.
        """
        self._conn.execute("""
            INSERT INTO session_log
                (session_id, query, query_vector, tool_used, plugin_id,
                 agent_id, agent_type, event, transcript_path)
            VALUES (?, NULL, NULL, 'subagent', NULL, ?, ?, ?, ?)
        """, (session_id, agent_id, agent_type, event, transcript_path or None))
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
                  session_id: str = "",
                  cwd: str = "", branch: str = "",
                  worktree: str = "", color: str = "",
                  repo: str = "") -> int:
        cur = self._conn.execute("""
            INSERT INTO tasks (title, summary, context, tags, vector,
                               session_id, cwd, branch, worktree, color,
                               node_id, repo)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (title, summary, context, tags, json.dumps(vector),
              session_id, cwd, branch, worktree or None, color or None,
              self.node_id, repo or None))
        self._conn.commit()
        task_id = cur.lastrowid or 0
        self._mirror_task_vec(task_id, vector)
        return task_id

    def set_task_worktree(self, task_id: int, worktree: str) -> bool:
        """Persist or refresh a task's worktree spec JSON. Empty string clears."""
        cur = self._conn.execute(
            "UPDATE tasks SET worktree = ?, updated_at = datetime('now') "
            "WHERE id = ?",
            (worktree or None, task_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def update_task(self, task_id: int, summary: str = "",
                    context: str = "", tags: str = "",
                    vector: list[float] | None = None,
                    title: str = "", color: str = "") -> bool:
        parts: list[str] = ["updated_at = datetime('now')"]
        params: list = []
        if title:
            parts.append("title = ?")
            params.append(title)
        if summary:
            parts.append("summary = ?")
            params.append(summary)
        if context:
            parts.append("context = ?")
            params.append(context)
        if tags:
            parts.append("tags = ?")
            params.append(tags)
        if color:
            parts.append("color = ?")
            params.append(color)
        if vector is not None:
            parts.append("vector = ?")
            params.append(json.dumps(vector))
        params.append(task_id)
        cur = self._conn.execute(
            f"UPDATE tasks SET {', '.join(parts)} WHERE id = ?", params
        )
        self._conn.commit()
        if vector is not None and cur.rowcount > 0:
            self._mirror_task_vec(task_id, vector)
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
        if compact_vector is not None and cur.rowcount > 0:
            self._mirror_task_vec(task_id, compact_vector)
        return cur.rowcount > 0

    def _mirror_task_vec(self, task_id: int, vector: list[float]) -> None:
        """Best-effort mirror of a task vector into vec0 tables (S6)."""
        if self._vec_engine != "sqlite-vec":
            return
        try:
            self._ensure_vec_tables(len(vector))
        except Exception as exc:  # noqa: BLE001
            _log.debug("vec0 ensure failed for task %d: %s", task_id, exc)
        if self._vec_dim is None or len(vector) != self._vec_dim:
            if self._vec_dim is not None and not self._vec_dim_warned:
                _log.warning(
                    "embedding dim mismatch: expected %d, got %d for task %d — skipping",
                    self._vec_dim, len(vector), task_id,
                )
                self._vec_dim_warned = True
            return
        try:
            self._write_secondary_vec_rows(
                task_id, vector, "task_id", "tasks_vec_bin", "tasks_vec_f32"
            )
        except Exception as exc:  # noqa: BLE001
            _log.debug("task vec0 mirror failed for %d: %s", task_id, exc)

    def _mirror_teaching_vec(self, teaching_id: int, vector: list[float]) -> None:
        """Best-effort mirror of a teaching rule vector into vec0 tables (S6)."""
        if self._vec_engine != "sqlite-vec":
            return
        try:
            self._ensure_vec_tables(len(vector))
        except Exception as exc:  # noqa: BLE001
            _log.debug("vec0 ensure failed for teaching %d: %s", teaching_id, exc)
        if self._vec_dim is None or len(vector) != self._vec_dim:
            if self._vec_dim is not None and not self._vec_dim_warned:
                _log.warning(
                    "embedding dim mismatch: expected %d, got %d for teaching %d — skipping",
                    self._vec_dim, len(vector), teaching_id,
                )
                self._vec_dim_warned = True
            return
        try:
            self._write_secondary_vec_rows(
                teaching_id, vector, "teaching_id",
                "teachings_vec_bin", "teachings_vec_f32",
            )
        except Exception as exc:  # noqa: BLE001
            _log.debug("teaching vec0 mirror failed for %d: %s", teaching_id, exc)

    def reopen_task(self, task_id: int) -> bool:
        cur = self._conn.execute("""
            UPDATE tasks SET status = 'open', closed_at = NULL,
                             updated_at = datetime('now')
            WHERE id = ?
        """, (task_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def list_tasks(self, status: str = "open",
                   tag: str | None = None,
                   repo: str | None = None,
                   worktree_path: str | None = None,
                   branch: str | None = None) -> list[sqlite3.Row]:
        cols = (
            "id, title, summary, context, status, tags, color, session_id, "
            "repo, cwd, branch, created_at, updated_at, closed_at"
        )
        where: list[str] = []
        params: list = []
        if status != "all":
            where.append("status = ?")
            params.append(status)
        if tag:
            # Tags are stored as a space- or comma-delimited string; match the
            # token bounded by start/end or non-word chars to avoid `fanout:abc`
            # matching `fanout:abcdef`.
            where.append(
                "(tags = ? OR tags LIKE ? OR tags LIKE ? OR tags LIKE ?)"
            )
            params += [tag, f"{tag} %", f"% {tag}", f"% {tag} %"]
        if repo:
            where.append("repo = ?")
            params.append(repo)
        if worktree_path:
            # M1 #11 -- "tasks belonging to this worktree". cwd is the
            # canonical column (auto-captured by save_task from
            # `git rev-parse --show-toplevel`).
            where.append("cwd = ?")
            params.append(worktree_path)
        if branch:
            where.append("branch = ?")
            params.append(branch)
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        return self._conn.execute(
            f"SELECT {cols} FROM tasks{clause} ORDER BY updated_at DESC",
            params,
        ).fetchall()

    def find_open_tasks_by_branch(self, branch: str,
                                  repo: str | None = None) -> list[sqlite3.Row]:
        """Return all open tasks recorded against ``branch``.

        Used by the optional post-merge git hook (M1 #11) to close tasks
        when their branch is deleted. ``repo`` scopes the lookup so two
        repositories with a same-named branch never collide.
        """
        if not branch:
            return []
        params: list = [branch]
        clause = ""
        if repo:
            clause = " AND repo = ?"
            params.append(repo)
        return self._conn.execute(
            f"SELECT * FROM tasks WHERE status = 'open' AND branch = ?{clause} "
            "ORDER BY updated_at DESC",
            params,
        ).fetchall()

    def list_tasks_by_repo(self, status: str = "open") -> dict[str, list[sqlite3.Row]]:
        """Group tasks by repo for dashboard rendering.

        Tasks with no repo land under the ``""`` (empty string) bucket so the
        caller can decide whether to label them ``"(unassigned)"``.
        """
        out: dict[str, list[sqlite3.Row]] = {}
        for row in self.list_tasks(status=status):
            try:
                key = row["repo"] or ""
            except (IndexError, KeyError):
                key = ""
            out.setdefault(key, []).append(row)
        return out

    def get_task(self, task_id: int) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()

    # ------------------------------------------------------------------
    # Issue #37 — typed task↔issue link store methods
    # ------------------------------------------------------------------

    def link_task_issue(
        self,
        task_id: int,
        issue_number: int,
        repo: str = "",
        url: str | None = None,
    ) -> None:
        """Create a typed link between a task and a GitHub issue (INSERT OR IGNORE)."""
        self._conn.execute(
            "INSERT OR IGNORE INTO task_issue_links "
            "(task_id, repo, issue_number, url) VALUES (?, ?, ?, ?)",
            (task_id, repo or "", issue_number, url),
        )
        self._conn.commit()

    def get_issue_links(self, task_id: int) -> list[dict]:
        """Return all issue links for a task as plain dicts."""
        rows = self._conn.execute(
            "SELECT * FROM task_issue_links WHERE task_id = ? ORDER BY id",
            (task_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_all_issue_links(self, repo: str = "") -> list[dict]:
        """Return all issue link rows, optionally filtered by repo."""
        if repo:
            rows = self._conn.execute(
                "SELECT * FROM task_issue_links WHERE repo = ? ORDER BY id",
                (repo,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM task_issue_links ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]

    def update_link_state(
        self,
        link_id: int,
        state: str,
        *,
        writeback_done: int | None = None,
    ) -> None:
        """Update the last-known issue state (and optionally writeback_done) for a link."""
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if writeback_done is not None:
            self._conn.execute(
                "UPDATE task_issue_links SET state = ?, last_synced_at = ?, "
                "writeback_done = ? WHERE id = ?",
                (state, now, writeback_done, link_id),
            )
        else:
            self._conn.execute(
                "UPDATE task_issue_links SET state = ?, last_synced_at = ? WHERE id = ?",
                (state, now, link_id),
            )
        self._conn.commit()

    def get_open_task_for_session(self, session_id: str) -> sqlite3.Row | None:
        """Return the most recently created open task for a session, if any."""
        if not session_id:
            return None
        return self._conn.execute(
            "SELECT * FROM tasks WHERE status = 'open' AND session_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (session_id,)
        ).fetchone()

    def get_open_task_id_for_session(self, session_id: str) -> int | None:
        """Return the id of the most recent open task for this session, or None."""
        if not session_id:
            return None
        row = self._conn.execute(
            "SELECT id FROM tasks WHERE session_id = ? AND status = 'open' "
            "ORDER BY created_at DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        if not row:
            return None
        return int(row["id"] if isinstance(row, dict) else row[0])

    _CLAUDE_COMPLETION_STATUSES = frozenset(
        {"completed", "done", "complete", "cancelled", "canceled", "stopped"}
    )

    def project_claude_task(
        self,
        *,
        key: str,
        title: str,
        status: str,
        claude_id: str | None = None,
        session_id: str = "",
        cwd: str = "",
        branch: str = "",
        summary: str = "",
    ) -> dict:
        """Upsert a skill-hub task row keyed by a stable Claude task hash.

        Returns a dict with keys ``action`` (``"created"``, ``"updated"``,
        ``"closed"``, ``"noop"``) and ``task_id`` (int or None).
        """
        row = self._conn.execute(
            "SELECT id, status, title, summary FROM tasks WHERE claude_task_key = ?",
            (key,),
        ).fetchone()

        if status in self._CLAUDE_COMPLETION_STATUSES:
            if row is not None and row["status"] == "open":
                tid = int(row["id"])
                self.close_task(tid, compact="auto-closed: Claude task completed")
                return {"action": "closed", "task_id": tid}
            return {"action": "noop", "task_id": row["id"] if row else None}

        # Open / in-progress / pending path.
        if row is not None:
            tid = int(row["id"])
            changed = (title and title != row["title"]) or (
                summary and summary != row["summary"]
            )
            if changed:
                self.update_task(tid, title=title or "", summary=summary or "")
            else:
                self.touch_task_activity(tid)
            return {"action": "updated", "task_id": tid}

        # Create new task.
        tid = self.save_task(
            title=title or "(untitled Claude task)",
            summary=summary,
            vector=[],
            tags="src:claude-task",
            session_id=session_id,
            cwd=cwd,
            branch=branch,
        )
        self._conn.execute(
            "UPDATE tasks SET claude_task_key = ?, claude_task_id = ? WHERE id = ?",
            (key, claude_id, tid),
        )
        self._conn.commit()
        return {"action": "created", "task_id": tid}

    # Cosine threshold for treating two tasks as "same work" during vector
    # dedup fallback.  0.85 was chosen because:
    #   - 0.90+ is near-duplicate text (too strict; misses paraphrases)
    #   - 0.80-  is topically related but not the same task (too loose)
    #   - 0.85 sits in the "same task, different wording" sweet spot seen in
    #     sentence-transformer benchmarks for short task titles.
    MEMORY_DEDUP_SIM_THRESHOLD = 0.85

    def project_memory_task(
        self,
        *,
        key: str,
        title: str,
        summary: str = "",
        tags: str = "",
        color: str = "",
        vector: list[float] | None = None,
        close: bool = False,
    ) -> dict:
        """Upsert a skill-hub task row sourced from a MEMORY.md entry.

        Dedup order:
        1. Exact stable-key match (``claude_task_key = key``).
        2. Vector-similarity fallback at ``MEMORY_DEDUP_SIM_THRESHOLD`` for
           tasks that pre-date stable-key support (no ``claude_task_key``).

        When ``close=True`` the matched / created task is immediately closed
        (used when the MEMORY.md entry is detected as SHIPPED/DONE).

        Returns a dict with ``action`` (``"created"``, ``"updated"``,
        ``"closed"``, ``"noop"``) and ``task_id``.
        """
        vec = vector or []

        # --- Tier 1: exact stable-key match ---
        row = self._conn.execute(
            "SELECT id, status, title, summary FROM tasks WHERE claude_task_key = ?",
            (key,),
        ).fetchone()

        if row is None and vec:
            # --- Tier 2: vector-similarity fallback (no window restriction) ---
            hit = self._find_open_task_by_vector(vec, self.MEMORY_DEDUP_SIM_THRESHOLD)
            if hit is not None:
                existing_row, _score = hit
                # Adopt this pre-existing task: stamp the stable key so future
                # runs skip the similarity scan.
                self._conn.execute(
                    "UPDATE tasks SET claude_task_key = ? WHERE id = ?",
                    (key, existing_row["id"]),
                )
                self._conn.commit()
                row = self._conn.execute(
                    "SELECT id, status, title, summary FROM tasks WHERE id = ?",
                    (existing_row["id"],),
                ).fetchone()

        if close:
            if row is not None and row["status"] == "open":
                tid = int(row["id"])
                self.close_task(tid, compact="auto-closed: memory entry marked done")
                return {"action": "closed", "task_id": tid}
            return {"action": "noop", "task_id": row["id"] if row else None}

        if row is not None:
            tid = int(row["id"])
            changed = (title and title != row["title"]) or (
                summary and summary != row["summary"]
            )
            if changed:
                kw: dict = {"title": title or "", "summary": summary or ""}
                if tags:
                    kw["tags"] = tags
                if color:
                    kw["color"] = color
                if vec:
                    kw["vector"] = vec
                self.update_task(tid, **kw)
            else:
                self.touch_task_activity(tid)
            return {"action": "updated", "task_id": tid}

        # Create new task.
        tid = self.save_task(
            title=title or "(untitled memory task)",
            summary=summary,
            vector=vec,
            context="",
            tags=tags or "src:memory",
            session_id="",
            color=color or "",
        )
        self._conn.execute(
            "UPDATE tasks SET claude_task_key = ? WHERE id = ?",
            (key, tid),
        )
        self._conn.commit()
        return {"action": "created", "task_id": tid}

    def _find_open_task_by_vector(
        self, query_vec: list[float], threshold: float
    ) -> tuple[sqlite3.Row, float] | None:
        """Return the top-1 open task whose cosine similarity to query_vec
        meets ``threshold``, searching across ALL time (no window).

        Returns None if no match or if query_vec is empty.
        """
        if not query_vec:
            return None
        import math
        rows = self._conn.execute(
            "SELECT * FROM tasks WHERE status = 'open' "
            "AND vector IS NOT NULL AND vector != '[]'"
        ).fetchall()
        if not rows:
            return None
        qn = math.sqrt(sum(v * v for v in query_vec)) or 1.0
        best_row, best_score = None, 0.0
        for r in rows:
            try:
                v = json.loads(r["vector"])
                if not v or len(v) != len(query_vec):
                    continue
                dot = sum(a * b for a, b in zip(query_vec, v))
                vn = math.sqrt(sum(x * x for x in v)) or 1.0
                score = dot / (qn * vn)
                if score > best_score:
                    best_row, best_score = r, score
            except (json.JSONDecodeError, TypeError):
                continue
        if best_row is not None and best_score >= threshold:
            return best_row, best_score
        return None

    def find_open_task_by_stable_key(self, key: str) -> sqlite3.Row | None:
        """Return the open task whose ``claude_task_key`` matches ``key``, or None."""
        if not key:
            return None
        return self._conn.execute(
            "SELECT * FROM tasks WHERE status = 'open' AND claude_task_key = ?",
            (key,),
        ).fetchone()

    def find_resumable_task_by_cwd_branch(
        self, cwd: str, branch: str, window_days: int
    ) -> sqlite3.Row | None:
        """Most recently updated open task matching cwd+branch within window, or None."""
        if not cwd:
            return None
        return self._conn.execute(
            "SELECT * FROM tasks WHERE status = 'open' "
            "AND cwd = ? AND IFNULL(branch,'') = ? "
            "AND updated_at > datetime('now', ?) "
            "ORDER BY updated_at DESC LIMIT 1",
            (cwd, branch or "", f"-{int(window_days)} days"),
        ).fetchone()

    def find_resumable_task_semantic(
        self, query_vec: list[float], window_days: int, threshold: float
    ) -> tuple[sqlite3.Row, float] | None:
        """Top-1 open task within window whose stored vector has cosine >= threshold.

        Best-effort: returns None if no rows, no stored vectors, or below threshold.
        """
        if not query_vec:
            return None
        import math
        rows = self._conn.execute(
            "SELECT * FROM tasks WHERE status = 'open' "
            "AND updated_at > datetime('now', ?) "
            "AND vector IS NOT NULL AND vector != '[]'",
            (f"-{int(window_days)} days",),
        ).fetchall()
        if not rows:
            return None
        qn = math.sqrt(sum(v * v for v in query_vec)) or 1.0
        best_row, best_score = None, 0.0
        for r in rows:
            try:
                v = json.loads(r["vector"])
                if not v or len(v) != len(query_vec):
                    continue
                dot = sum(a * b for a, b in zip(query_vec, v))
                vn = math.sqrt(sum(x * x for x in v)) or 1.0
                score = dot / (qn * vn)
                if score > best_score:
                    best_row, best_score = r, score
            except (json.JSONDecodeError, TypeError):
                continue
        if best_row is not None and best_score >= threshold:
            return best_row, best_score
        return None

    def bind_task_to_session(self, task_id: int, session_id: str) -> None:
        """Rebind an existing open task to a new session; bump updated_at + last_activity_at.

        Writes both sides of the session<->task link: ``tasks.session_id``
        and, when a ``session_context`` row already exists for
        ``session_id``, ``session_context.task_id`` (see #127).
        """
        self._conn.execute(
            "UPDATE tasks SET session_id = ?, "
            "updated_at = datetime('now'), last_activity_at = datetime('now') "
            "WHERE id = ?",
            (session_id, task_id),
        )
        self._conn.execute(
            "UPDATE session_context SET task_id = ? WHERE session_id = ?",
            (task_id, session_id),
        )
        self._conn.commit()

    # Title shape produced by the pre-#127 auto-create bug in
    # session_start_enforcer._ensure_open_tasks: a bare filename slug
    # (lowercase/digits joined by underscores) used verbatim as the task
    # title instead of the memory file's human-readable description.
    _JUNK_MEMORY_TITLE_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)+$")

    def find_junk_memory_tasks(self) -> list[sqlite3.Row]:
        """Return ``src:memory`` tasks with no session that look auto-created
        with a raw filename slug as their title (see #127).

        ``project_memory_task`` always writes ``session_id=""`` (never SQL
        NULL) via ``save_task``'s default, so both representations are
        matched defensively.
        """
        rows = self._conn.execute(
            "SELECT id, title, tags, session_id FROM tasks "
            "WHERE (session_id IS NULL OR session_id = '') "
            "AND (tags = 'src:memory' OR tags LIKE 'src:memory,%' "
            "OR tags LIKE '%,src:memory' OR tags LIKE '%,src:memory,%')"
        ).fetchall()
        return [
            r for r in rows
            if self._JUNK_MEMORY_TITLE_RE.match((r["title"] or "").strip())
        ]

    def cleanup_junk_memory_tasks(self, *, dry_run: bool = True) -> dict:
        """Idempotently remove junk auto-created memory tasks (see #127).

        Safe to run repeatedly: once the matching rows are gone this is a
        no-op. Returns ``{"dry_run", "removed": [{"id", "title"}], "count"}``.
        """
        junk = self.find_junk_memory_tasks()
        removed = [{"id": int(r["id"]), "title": r["title"]} for r in junk]
        if not dry_run and junk:
            self._conn.executemany(
                "DELETE FROM tasks WHERE id = ?", [(r["id"],) for r in junk]
            )
            self._conn.commit()
        return {"dry_run": dry_run, "removed": removed, "count": len(removed)}

    # ─────────────────────── M1 claims layer ─────────────────────────────
    #
    # The four primitives below let multiple Claude Code sessions or swarm
    # subprocesses agree on who currently owns a task without an LLM in the
    # loop. They never block — every operation is either a single atomic
    # ``UPDATE ... WHERE`` with a status-precondition clause or a no-op.
    #
    # Contract:
    #   - A claim is FREE when ``claimed_by IS NULL``.
    #   - ``claim_task`` succeeds iff the row is currently free.
    #   - ``handoff_task`` succeeds iff the caller already holds the claim.
    #   - ``steal_task`` succeeds iff ``stealable_at <= now()``.
    #   - ``release_task`` succeeds iff the caller currently holds the claim
    #     (or ``agent_id`` is None, used for force-release in admin paths).
    # Each successful state transition issues a fresh ``claim_token`` so
    # racing callers cannot revive a stale lock.

    def claim_task(
        self,
        task_id: int,
        agent_id: str,
        stealable_after_sec: int | None = None,
    ) -> str | None:
        """Atomically claim a free task for ``agent_id``.

        Returns the new claim_token on success, or None when the task is
        already claimed / does not exist / is closed.
        """
        if not agent_id:
            return None
        token = uuid.uuid4().hex
        stealable_clause = ""
        params: list = [agent_id, token]
        if stealable_after_sec is not None and stealable_after_sec >= 0:
            stealable_clause = ", stealable_at = datetime('now', ?)"
            params.append(f"+{int(stealable_after_sec)} seconds")
        else:
            stealable_clause = ", stealable_at = NULL"
        params.append(task_id)
        cur = self._conn.execute(
            f"""
            UPDATE tasks
            SET claimed_by = ?, claim_token = ?,
                claimed_at = datetime('now'),
                updated_at = datetime('now')
                {stealable_clause}
            WHERE id = ? AND claimed_by IS NULL AND status = 'open'
            """,
            params,
        )
        self._conn.commit()
        return token if cur.rowcount > 0 else None

    def handoff_task(
        self,
        task_id: int,
        to_agent: str,
        from_agent: str | None = None,
        stealable_after_sec: int | None = None,
    ) -> str | None:
        """Transfer a claim from the current owner to ``to_agent``.

        When ``from_agent`` is supplied, the handoff only succeeds if it
        matches the current ``claimed_by``. When omitted, any caller who
        knows the task_id can hand off (intended for admin / hub paths).
        Returns the new claim_token on success, else None.
        """
        if not to_agent:
            return None
        token = uuid.uuid4().hex
        where = "id = ? AND claimed_by IS NOT NULL AND status = 'open'"
        params: list = [to_agent, token]
        stealable_clause = ", stealable_at = NULL"
        if stealable_after_sec is not None and stealable_after_sec >= 0:
            stealable_clause = ", stealable_at = datetime('now', ?)"
            params.append(f"+{int(stealable_after_sec)} seconds")
        params.append(task_id)
        if from_agent:
            where += " AND claimed_by = ?"
            params.append(from_agent)
        cur = self._conn.execute(
            f"""
            UPDATE tasks
            SET claimed_by = ?, claim_token = ?,
                claimed_at = datetime('now'),
                updated_at = datetime('now')
                {stealable_clause}
            WHERE {where}
            """,
            params,
        )
        self._conn.commit()
        return token if cur.rowcount > 0 else None

    def steal_task(
        self,
        task_id: int,
        new_agent_id: str,
        stealable_after_sec: int | None = None,
    ) -> str | None:
        """Forcefully take over a claim whose ``stealable_at`` has elapsed.

        Returns the new claim_token on success, else None (task free, not
        yet stealable, or missing).
        """
        if not new_agent_id:
            return None
        token = uuid.uuid4().hex
        stealable_clause = ", stealable_at = NULL"
        params: list = [new_agent_id, token]
        if stealable_after_sec is not None and stealable_after_sec >= 0:
            stealable_clause = ", stealable_at = datetime('now', ?)"
            params.append(f"+{int(stealable_after_sec)} seconds")
        params.append(task_id)
        cur = self._conn.execute(
            f"""
            UPDATE tasks
            SET claimed_by = ?, claim_token = ?,
                claimed_at = datetime('now'),
                updated_at = datetime('now')
                {stealable_clause}
            WHERE id = ?
              AND claimed_by IS NOT NULL
              AND status = 'open'
              AND stealable_at IS NOT NULL
              AND stealable_at <= datetime('now')
            """,
            params,
        )
        self._conn.commit()
        return token if cur.rowcount > 0 else None

    def release_task(
        self,
        task_id: int,
        agent_id: str | None = None,
    ) -> bool:
        """Clear the claim on a task.

        When ``agent_id`` is supplied, only release if it matches the
        current ``claimed_by`` (prevents accidental cross-agent release).
        Returns True on a real state change, False otherwise (already
        free, wrong agent, or missing task).
        """
        params: list = [task_id]
        where = "id = ? AND claimed_by IS NOT NULL"
        if agent_id:
            where += " AND claimed_by = ?"
            params.append(agent_id)
        cur = self._conn.execute(
            f"""
            UPDATE tasks
            SET claimed_by = NULL, claim_token = NULL,
                claimed_at = NULL, stealable_at = NULL,
                updated_at = datetime('now')
            WHERE {where}
            """,
            params,
        )
        self._conn.commit()
        return cur.rowcount > 0

    def get_task_claim(self, task_id: int) -> dict | None:
        """Return the current claim metadata for a task, or None if missing.

        Always returns a dict with all four claim columns (any of which may
        be None) so callers can pattern-match on `claimed_by is None`.
        """
        row = self._conn.execute(
            "SELECT claimed_by, claim_token, claimed_at, stealable_at "
            "FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "claimed_by":   row["claimed_by"],
            "claim_token":  row["claim_token"],
            "claimed_at":   row["claimed_at"],
            "stealable_at": row["stealable_at"],
        }

    def touch_task_activity(self, task_id: int) -> None:
        """Update last_activity_at = now() for a task. Best-effort, never raises."""
        try:
            self._conn.execute(
                "UPDATE tasks SET last_activity_at = datetime('now') WHERE id = ?",
                (task_id,),
            )
            self._conn.commit()
        except Exception:
            pass

    def get_task_activity_state(self, task_id: int) -> str:
        """Return 'active'|'idle'|'open'|'closed' based on last_activity_at."""
        row = self._conn.execute(
            "SELECT status, last_activity_at FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if not row:
            return "unknown"
        status = row["status"] if isinstance(row, dict) else row[0]
        last_at = row["last_activity_at"] if isinstance(row, dict) else row[1]
        if status == "closed":
            return "closed"
        if not last_at:
            return "open"
        from datetime import datetime, timezone
        try:
            dt = datetime.fromisoformat(last_at.replace("Z", "+00:00"))
            # SQLite datetime('now') is UTC but has no tzinfo — normalise to aware UTC.
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            diff_sec = (now - dt).total_seconds()
            if diff_sec <= 60:
                return "active"
            if diff_sec <= 3600:
                return "idle"
        except Exception:
            pass
        return "open"

    def list_tasks_with_activity(self, status: str | None = None, limit: int = 50) -> list[dict]:
        """Return tasks with computed activity_state column."""
        from datetime import datetime, timezone

        where = "WHERE status = ?" if status else ""
        params: tuple = (status,) if status else ()
        rows = self._conn.execute(
            f"SELECT id, title, summary, status, tags, session_id, "
            f"created_at, updated_at, last_activity_at, compact "
            f"FROM tasks {where} ORDER BY "
            f"CASE status WHEN 'open' THEN 0 ELSE 1 END, "
            f"updated_at DESC LIMIT ?",
            (*params, limit),
        ).fetchall()

        def _state(row) -> str:
            if isinstance(row, dict):
                st = row.get("status")
                la = row.get("last_activity_at")
            else:
                # sqlite3.Row — access by column name
                try:
                    st = row["status"]
                    la = row["last_activity_at"]
                except (IndexError, KeyError):
                    return "open"
            if st == "closed":
                return "closed"
            if not la:
                return "open"
            try:
                dt = datetime.fromisoformat(la.replace("Z", "+00:00"))
                # SQLite datetime('now') is UTC but has no tzinfo — normalise to aware UTC.
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                diff = (datetime.now(timezone.utc) - dt).total_seconds()
                if diff <= 60:
                    return "active"
                if diff <= 3600:
                    return "idle"
            except Exception:
                pass
            return "open"

        result = []
        for row in rows:
            d = dict(row)
            d["activity_state"] = _state(row)
            result.append(d)
        return result

    def set_task_auto_approve(self, task_id: int, enabled: bool | None) -> bool:
        """Set per-task auto_approve flag. None clears (global behavior).

        Delegates to set_task_options to keep options column in sync.
        """
        return self.set_task_options(task_id, {"auto_approve": enabled})

    def get_task_auto_approve(self, task_id: int) -> bool | None:
        row = self._conn.execute(
            "SELECT auto_approve, options FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if row is None:
            return None
        # options.auto_approve takes priority if set explicitly
        if row["options"]:
            try:
                opts = json.loads(row["options"])
                if "auto_approve" in opts and opts["auto_approve"] is not None:
                    return bool(opts["auto_approve"])
            except (TypeError, json.JSONDecodeError):
                pass
        if row["auto_approve"] is None:
            return None
        return bool(row["auto_approve"])

    def get_task_options(self, task_id: int) -> dict:
        """Return per-task options dict (merged from options column + auto_approve column).

        Keys: auto_approve, routing_disabled, model_pin, preload_skills, forbid_skills.
        All keys are optional; absence means "inherit global default".
        """
        row = self._conn.execute(
            "SELECT auto_approve, options FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if row is None:
            return {}
        opts: dict = {}
        if row["options"]:
            try:
                opts = json.loads(row["options"]) or {}
            except (TypeError, json.JSONDecodeError):
                opts = {}
        # Back-fill from the legacy auto_approve column when not in options.
        if "auto_approve" not in opts and row["auto_approve"] is not None:
            opts["auto_approve"] = bool(row["auto_approve"])
        return opts

    def set_task_options(self, task_id: int, patch: dict) -> bool:
        """Patch per-task options.  Keys with value None are removed.

        Example: set_task_options(1, {"routing_disabled": True}) enables the flag.
                 set_task_options(1, {"routing_disabled": None}) removes the key.
        """
        current = self.get_task_options(task_id)
        for k, v in patch.items():
            if v is None:
                current.pop(k, None)
            else:
                current[k] = v
        # Mirror auto_approve into the legacy column for backwards compat.
        aa = current.get("auto_approve")
        aa_val = None if aa is None else (1 if aa else 0)
        options_json = json.dumps(current) if current else None
        cur = self._conn.execute(
            "UPDATE tasks SET options = ?, auto_approve = ?, updated_at = datetime('now') "
            "WHERE id = ?",
            (options_json, aa_val, task_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def search_tasks(self, query_vector: list[float] | None = None, top_k: int = 3,
                     status: str = "all", min_sim: float = 0.4,
                     text_query: str | None = None) -> list[dict]:
        """Search tasks by semantic similarity or BM25 full-text search.

        When ``query_vector`` is None or empty and ``text_query`` is provided,
        falls back to FTS5 BM25 search (zero ML deps required).

        Uses the vec0 binary-KNN + float32 rerank path when available (S6);
        falls back to the in-Python cosine loop for legacy/short vectors.
        """
        # FTS5 fallback: no vector provided but text query given
        if not query_vector and text_query:
            fts_status = None if status == "all" else status
            return self.search_text(text_query, tables=["tasks"],
                                    top_k=top_k, status=fts_status)

        if query_vector is None:
            return []

        if self._vec_engine == "sqlite-vec" and self._vec_dim is not None and len(query_vector) == self._vec_dim:
            vec_results = self._search_tasks_vec(query_vector, top_k, status, min_sim)
            if vec_results is not None:
                return vec_results

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

    def _search_tasks_vec(self, query_vector: list[float], top_k: int,
                          status: str, min_sim: float) -> list[dict] | None:
        """S6 vec0 path for tasks. Returns None to signal fallback on failure."""
        from . import config as _cfg
        from .embeddings import quantize_binary

        rerank_k = max(top_k, int(_cfg.get("rerank_top_k") or 20))
        try:
            qbin = quantize_binary(query_vector)
            bin_rows = self._conn.execute(
                """
                SELECT task_id, distance
                FROM tasks_vec_bin
                WHERE embedding MATCH vec_bit(?) AND k = ?
                ORDER BY distance
                """,
                (qbin, rerank_k),
            ).fetchall()
        except Exception:
            return None
        if not bin_rows:
            return []
        ids = [r["task_id"] for r in bin_rows]

        qnorm = math.sqrt(sum(x * x for x in query_vector))
        if qnorm == 0.0:
            return []

        placeholders = ",".join("?" * len(ids))
        where = "" if status == "all" else f"AND t.status = '{status}'"
        rows = self._conn.execute(
            f"""
            SELECT t.id, t.title, t.summary, t.context, t.status,
                   t.tags, t.compact, t.vector, t.created_at, t.closed_at
            FROM tasks t
            WHERE t.id IN ({placeholders}) AND t.vector IS NOT NULL {where}
            """,
            ids,
        ).fetchall()

        scored: list[tuple[float, dict]] = []
        for row in rows:
            try:
                vec = json.loads(row["vector"])
            except (TypeError, ValueError):
                continue
            snorm = math.sqrt(sum(x * x for x in vec))
            if snorm == 0.0:
                continue
            dot = sum(a * b for a, b in zip(query_vector, vec))
            sim = dot / (qnorm * snorm)
            if sim < min_sim:
                continue
            d = dict(row)
            d["similarity"] = sim
            scored.append((sim, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def delete_task(self, task_id: int) -> bool:
        cur = self._conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        self._conn.commit()
        if self._vec_engine == "sqlite-vec":
            try:
                self._conn.execute("DELETE FROM tasks_vec_bin WHERE task_id = ?", (task_id,))
                self._conn.execute("DELETE FROM tasks_vec_f32 WHERE task_id = ?", (task_id,))
                self._conn.commit()
            except Exception:
                pass
        return cur.rowcount > 0

    def rename_task_title(self, task_id: int, title: str) -> bool:
        cur = self._conn.execute(
            "UPDATE tasks SET title = ?, updated_at = datetime('now') WHERE id = ?",
            (title, task_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def merge_tasks(self, task_ids: list[int]) -> int:
        """Concatenate summaries into a new open task; mark originals closed.

        Returns the new task id (0 on failure).
        """
        if not task_ids:
            return 0
        rows = self._conn.execute(
            f"SELECT id, title, summary, context, tags FROM tasks "
            f"WHERE id IN ({','.join('?' * len(task_ids))})",
            task_ids,
        ).fetchall()
        if not rows:
            return 0
        title = "Merged: " + " + ".join(r["title"][:40] for r in rows)
        parts = []
        ctx_parts = []
        tags: list[str] = []
        for r in rows:
            parts.append(f"## Task #{r['id']}: {r['title']}\n{r['summary']}")
            if r["context"]:
                ctx_parts.append(f"### From #{r['id']}\n{r['context']}")
            if r["tags"]:
                tags.append(r["tags"])
        summary = "\n\n".join(parts)
        context = "\n\n".join(ctx_parts)
        merged_tags = ",".join(sorted({t.strip() for s in tags for t in s.split(",") if t.strip()}))
        cur = self._conn.execute(
            "INSERT INTO tasks (title, summary, context, tags, vector, status) "
            "VALUES (?, ?, ?, ?, NULL, 'open')",
            (title[:200], summary, context, merged_tags),
        )
        new_id = cur.lastrowid or 0
        # Mark originals closed pointing to new task
        for tid in task_ids:
            self._conn.execute(
                "UPDATE tasks SET status='closed', closed_at=datetime('now'), "
                "compact=?, updated_at=datetime('now') WHERE id = ?",
                (json.dumps({"merged_into": new_id}), tid),
            )
        self._conn.commit()
        return new_id

    def search_tasks_text(self, query: str, status: str = "all") -> list[sqlite3.Row]:
        pattern = f"%{query}%"
        where_status = "" if status == "all" else "AND status = ?"
        params: list = [pattern, pattern, pattern]
        if status != "all":
            params.append(status)
        return self._conn.execute(
            f"SELECT id, title, summary, status, tags, created_at, updated_at, "
            f"closed_at FROM tasks "
            f"WHERE (title LIKE ? OR summary LIKE ? OR tags LIKE ?) {where_status} "
            f"ORDER BY updated_at DESC LIMIT 100",
            params,
        ).fetchall()

    # ------------------------------------------------------------------
    # Pipeline telemetry

    def record_pipeline_run(
        self,
        session_id: str,
        task_id: int | None,
        tier_ms: dict[str, int | None],
        fallbacks: list[str],
        top_similarity: float | None = None,
        token_cost_usd: float = 0.0,
    ) -> int:
        """Insert one telemetry row for a completed pipeline run.

        Returns the rowid of the inserted record.
        """
        cur = self._conn.execute(
            """INSERT INTO pipeline_runs
               (session_id, task_id, tier1_ms, tier2_ms, tier3_ms, tier4_ms,
                fallbacks_used, top_similarity, token_cost_usd)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                task_id,
                tier_ms.get("tier1"),
                tier_ms.get("tier2"),
                tier_ms.get("tier3"),
                tier_ms.get("tier4"),
                json.dumps(fallbacks),
                top_similarity,
                token_cost_usd,
            ),
        )
        self._conn.commit()
        return cur.lastrowid or 0

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Sanitize a user query string for safe use in FTS5 MATCH expressions.

        FTS5 special characters that must be escaped or stripped: " * ( ) - ^
        Strategy: strip characters that have no safe literal equivalent, then
        wrap remaining terms to avoid accidental prefix/phrase operators.
        Boolean operators (AND, OR, NOT) are also stripped so SQLite FTS5 does
        not interpret them as query syntax.
        """
        # Remove characters that are FTS5 operators or cause parse errors
        cleaned = re.sub(r'["\*\(\)\-\^]', ' ', query)
        # Strip FTS5 boolean operators so they are not interpreted as syntax
        cleaned = re.sub(r'\b(AND|OR|NOT)\b', ' ', cleaned)
        # Collapse whitespace
        cleaned = ' '.join(cleaned.split())
        return cleaned if cleaned else '""'

    def search_text(
        self,
        query: str,
        tables: list[str] | None = None,
        top_k: int = 10,
        status: str | None = None,
    ) -> list[dict]:
        """BM25 full-text search via SQLite FTS5. Always available, zero ML deps.

        Returns list of dicts with keys: id, type (tasks/teachings),
        title_or_rule, summary_or_why, score, status (for tasks only).

        Falls back to an empty list gracefully if FTS5 tables are unavailable.
        """
        if not query or not query.strip():
            return []

        safe_query = self._sanitize_fts_query(query)
        if not safe_query or safe_query == '""':
            return []

        search_tables = tables if tables is not None else ["tasks", "teachings"]
        results: list[dict] = []

        # --- tasks ---
        if "tasks" in search_tables:
            try:
                if status is not None:
                    rows = self._conn.execute(
                        """
                        SELECT f.rowid AS id, f.rank AS score,
                               t.title, t.summary, t.status
                        FROM tasks_fts f
                        JOIN tasks t ON t.id = f.rowid
                        WHERE tasks_fts MATCH ? AND t.status = ?
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (safe_query, status, top_k),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        """
                        SELECT f.rowid AS id, f.rank AS score,
                               t.title, t.summary, t.status
                        FROM tasks_fts f
                        JOIN tasks t ON t.id = f.rowid
                        WHERE tasks_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (safe_query, top_k),
                    ).fetchall()
                for row in rows:
                    results.append({
                        "id": row["id"],
                        "type": "tasks",
                        "title_or_rule": row["title"],
                        "summary_or_why": row["summary"],
                        "score": float(row["score"]),
                        "status": row["status"],
                    })
            except sqlite3.OperationalError as exc:
                if "no such table" in str(exc).lower():
                    _log.debug("FTS5 tasks table not yet created: %s", exc)
                else:
                    raise

        # --- teachings ---
        if "teachings" in search_tables:
            try:
                rows = self._conn.execute(
                    """
                    SELECT f.rowid AS id, f.rank AS score,
                           t.rule, t.action
                    FROM teachings_fts f
                    JOIN teachings t ON t.id = f.rowid
                    WHERE teachings_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (safe_query, top_k),
                ).fetchall()
                for row in rows:
                    results.append({
                        "id": row["id"],
                        "type": "teachings",
                        "title_or_rule": row["rule"],
                        "summary_or_why": row["action"],
                        "score": float(row["score"]),
                        "status": None,
                    })
            except sqlite3.OperationalError as exc:
                if "no such table" in str(exc).lower():
                    _log.debug("FTS5 teachings table not yet created: %s", exc)
                else:
                    raise

        # BM25 rank is negative in SQLite FTS5 — lower (more negative) = better match.
        # Sort so best matches come first (ascending by score = most negative first).
        results.sort(key=lambda x: x["score"])
        return results[:top_k]

    def search_skills_text(self, query: str, top_k: int = 5,
                           target: str | None = None) -> list[dict]:
        """BM25 keyword fallback for skill search when embeddings are unavailable.

        Returns rows shaped like :meth:`search` results so callers can swap the
        two paths without restructuring their result rendering. Empty list on
        empty query, no matches, or missing FTS5 table.
        """
        if not query or not query.strip():
            return []
        safe_query = self._sanitize_fts_query(query)
        if not safe_query or safe_query == '""':
            return []
        try:
            if target:
                rows = self._conn.execute(
                    """
                    SELECT s.id, s.name, s.description, s.content, s.plugin,
                           s.target, s.feedback_score, f.rank AS score
                    FROM skills_fts f
                    JOIN skills s ON s.id = f.skill_id
                    WHERE skills_fts MATCH ? AND s.target = ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (safe_query, target, top_k),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT s.id, s.name, s.description, s.content, s.plugin,
                           s.target, s.feedback_score, f.rank AS score
                    FROM skills_fts f
                    JOIN skills s ON s.id = f.skill_id
                    WHERE skills_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (safe_query, top_k),
                ).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                _log.debug("FTS5 skills table not yet created: %s", exc)
                return []
            raise

        return [
            {
                "id": r["id"],
                "name": r["name"],
                "description": r["description"],
                "content": r["content"],
                "plugin": r["plugin"],
                "target": r["target"],
                "feedback_score": r["feedback_score"],
                "score": float(r["score"]),
            }
            for r in rows
        ]

    def suggest_plugins_text(self, query: str, top_k: int = 5) -> list[dict]:
        """BM25 keyword fallback for plugin suggestions when embeddings are unavailable.

        Returns rows shaped like :meth:`suggest_plugins` results so callers can
        swap the two paths without restructuring rendering. ``embed_score`` is
        derived from the (negated, clipped) BM25 rank so it sorts the same way.
        """
        if not query or not query.strip():
            return []
        safe_query = self._sanitize_fts_query(query)
        if not safe_query or safe_query == '""':
            return []
        try:
            rows = self._conn.execute(
                """
                SELECT p.id, p.short_name, p.description, f.rank AS score
                FROM plugins_fts f
                JOIN plugins p ON p.id = f.plugin_id
                WHERE plugins_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (safe_query, top_k),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                _log.debug("FTS5 plugins table not yet created: %s", exc)
                return []
            raise

        # Convert BM25 rank (more negative = better) into a non-negative
        # embed_score-like value so the existing shape stays comparable.
        results: list[dict] = []
        for r in rows:
            # Clip and invert so best-match -> highest score; keeps it loosely
            # in the 0..1 range typical of cosine similarity, without
            # over-promising semantic precision.
            bm25 = float(r["score"])
            pseudo = max(0.0, min(1.0, -bm25 / 10.0))
            results.append({
                "plugin_id": r["id"],
                "short_name": r["short_name"],
                "description": r["description"],
                "embed_score": pseudo,
                "teaching_score": 0.0,
                "session_score": 0.0,
                "total_score": pseudo,
            })
        return results

    def get_skill_usage_stats(self) -> list[dict]:
        """Aggregate skill usage.

        Injections come from skill_injections (one row per search_skills hit).
        Helpful/unhelpful counts and last_used come from the feedback table.
        """
        rows = self._conn.execute("""
            SELECT s.id, s.name, s.plugin, s.target,
                   COALESCE(s.feedback_score, 1.0) as feedback_score,
                   COALESCE(inj.injections, 0) as injections,
                   COALESCE(fb.helpful, 0) as helpful,
                   COALESCE(fb.unhelpful, 0) as unhelpful,
                   COALESCE(inj.last_used, fb.last_used) as last_used
            FROM skills s
            LEFT JOIN (
                SELECT skill_id,
                       COUNT(*) as injections,
                       MAX(created_at) as last_used
                FROM skill_injections
                GROUP BY skill_id
            ) inj ON inj.skill_id = s.id
            LEFT JOIN (
                SELECT skill_id,
                       SUM(helpful) as helpful,
                       SUM(1 - helpful) as unhelpful,
                       MAX(created_at) as last_used
                FROM feedback
                GROUP BY skill_id
            ) fb ON fb.skill_id = s.id
            ORDER BY injections DESC, feedback_score DESC
            LIMIT 200
        """).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            tot = (d["helpful"] or 0) + (d["unhelpful"] or 0)
            d["helpful_pct"] = round(100 * (d["helpful"] or 0) / tot, 1) if tot else None
            out.append(d)
        return out

    def get_skills_for_task(self, task_id: int) -> list[dict]:
        """Return skills referenced during the task's session.

        Joins tasks.session_id -> session_log.tool_used (MCP tool) best-effort,
        plus any feedback rows recorded close to the task window.
        """
        row = self._conn.execute(
            "SELECT session_id, created_at, closed_at FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if not row:
            return []
        sid = row["session_id"]
        if not sid:
            return []
        rows = self._conn.execute("""
            SELECT tool_used, plugin_id, COUNT(*) as n
            FROM session_log
            WHERE session_id = ?
            GROUP BY tool_used, plugin_id
            ORDER BY n DESC
        """, (sid,)).fetchall()
        return [dict(r) for r in rows]

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
                "transcript_offset": 0,
                "task_id": None,
            }
        import json as _json
        return {
            "session_id": row["session_id"],
            "loaded_skills": _json.loads(row["loaded_skills"]),
            "context_summary": row["context_summary"],
            "message_count": row["message_count"],
            "recent_messages": _json.loads(row["recent_messages"]),
            "transcript_offset": row["transcript_offset"] if "transcript_offset" in row.keys() else 0,
            "task_id": row["task_id"] if "task_id" in row.keys() else None,
        }

    def save_session_context(self, session_id: str, loaded_skills: list[str],
                             context_summary: str, message_count: int,
                             recent_messages: list[str] | None = None) -> None:
        """Upsert the session context after dynamic evaluation.

        Also writes a plain-text file at SESSION_CONTEXT_FILE so local
        LLMs (L3 skills, L4 agent) can read conversation context from
        disk at zero Claude token cost.
        """
        import json as _json
        msgs_json = _json.dumps(recent_messages or [])
        # task_id is self-healed from tasks.session_id on first insert (the
        # row may not exist yet when bind_task_to_session runs on the very
        # first prompt — see #127) and left untouched on conflict so it
        # never regresses an already-bound value.
        self._conn.execute("""
            INSERT INTO session_context
                (session_id, loaded_skills, context_summary, message_count,
                 recent_messages, updated_at, task_id)
            VALUES (?, ?, ?, ?, ?, datetime('now'), (
                SELECT id FROM tasks WHERE tasks.session_id = ?
                AND tasks.status = 'open'
                ORDER BY tasks.created_at DESC LIMIT 1
            ))
            ON CONFLICT(session_id) DO UPDATE SET
                loaded_skills = excluded.loaded_skills,
                context_summary = excluded.context_summary,
                message_count = excluded.message_count,
                recent_messages = excluded.recent_messages,
                updated_at = excluded.updated_at
        """, (session_id, _json.dumps(loaded_skills), context_summary,
              message_count, msgs_json, session_id))
        self._conn.commit()

        # Write context file for local LLM consumption
        # Enrich with recent tool examples from DB + repo state
        tool_ex_data: list[dict] = []
        repo_ctx = ""
        try:
            rows = self.get_recent_tool_examples(limit=8)
            tool_ex_data = [dict(r) for r in rows]
        except Exception:
            pass
        try:
            import subprocess as _sp
            _rc = _sp.run(
                "git rev-parse --abbrev-ref HEAD 2>/dev/null && "
                "git diff --shortstat 2>/dev/null && "
                "git log --oneline -1 2>/dev/null",
                shell=True, capture_output=True, text=True, timeout=5,
            )
            if _rc.returncode == 0 and _rc.stdout.strip():
                repo_ctx = _rc.stdout.strip()
        except Exception:
            pass
        _write_session_context_file(
            context_summary=context_summary,
            recent_messages=recent_messages or [],
            tool_examples=tool_ex_data,
            repo_context=repo_ctx,
        )

    # ------------------------------------------------------------------
    # Context Bridge: tool examples + repo context

    def save_tool_example(self, session_id: str, tool_name: str,
                          tool_input: str, output_summary: str = "",
                          context_hint: str = "", repo_path: str = "",
                          category: str = "general") -> None:
        """Save a captured tool call example from Claude's transcript."""
        self._conn.execute("""
            INSERT INTO tool_examples
                (session_id, tool_name, tool_input, output_summary,
                 context_hint, repo_path, category)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_id, tool_name, tool_input[:500],
              output_summary[:200], context_hint[:300],
              repo_path, category))
        self._conn.commit()

    def save_tool_examples_batch(self, examples: list[dict]) -> None:
        """Batch-insert tool examples in a single transaction."""
        self._conn.executemany("""
            INSERT INTO tool_examples
                (session_id, tool_name, tool_input, output_summary,
                 context_hint, repo_path, category)
            VALUES (:session_id, :tool_name, :tool_input, :output_summary,
                    :context_hint, :repo_path, :category)
        """, examples)
        self._conn.commit()

    def get_recent_tool_examples(self, tool_name: str = "",
                                 repo_path: str = "",
                                 category: str = "",
                                 limit: int = 10) -> list[sqlite3.Row]:
        """Get recent tool examples, optionally filtered."""
        clauses: list[str] = []
        params: list = []
        if tool_name:
            clauses.append("tool_name = ?")
            params.append(tool_name)
        if repo_path:
            clauses.append("repo_path = ?")
            params.append(repo_path)
        if category:
            clauses.append("category = ?")
            params.append(category)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        return self._conn.execute(f"""
            SELECT tool_name, tool_input, output_summary, context_hint,
                   repo_path, category, created_at
            FROM tool_examples
            {where}
            ORDER BY created_at DESC
            LIMIT ?
        """, params).fetchall()

    def get_tool_patterns(self, limit: int = 20) -> list[dict]:
        """Aggregate tool usage patterns across sessions."""
        rows = self._conn.execute("""
            SELECT tool_name, repo_path, category, COUNT(*) as count,
                   GROUP_CONCAT(SUBSTR(tool_input, 1, 80), ' | ') as sample_inputs
            FROM (
                SELECT DISTINCT tool_name, repo_path, category, tool_input
                FROM tool_examples
            )
            GROUP BY tool_name, repo_path
            ORDER BY count DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def update_transcript_offset(self, session_id: str, offset: int) -> None:
        """Update the transcript processing offset for incremental parsing."""
        self._conn.execute("""
            UPDATE session_context SET transcript_offset = ?
            WHERE session_id = ?
        """, (offset, session_id))
        self._conn.commit()

    def prune_tool_examples(self, max_age_days: int = 30,
                            max_rows: int = 5000) -> int:
        """Prune old tool examples. Returns rows deleted."""
        cur = self._conn.execute("""
            DELETE FROM tool_examples
            WHERE created_at < datetime('now', '-' || ? || ' days')
        """, (max_age_days,))
        pruned = cur.rowcount
        # Also cap total rows
        cur2 = self._conn.execute("""
            DELETE FROM tool_examples
            WHERE id NOT IN (
                SELECT id FROM tool_examples
                ORDER BY created_at DESC LIMIT ?
            )
        """, (max_rows,))
        pruned += cur2.rowcount
        if pruned:
            self._conn.commit()
        return pruned

    def upsert_repo_context(self, repo_path: str, commit_style: str = "",
                            common_commands: str = "",
                            project_summary: str = "",
                            tool_stats: str = "") -> None:
        """Upsert accumulated per-repo knowledge."""
        self._conn.execute("""
            INSERT INTO repo_context
                (repo_path, commit_style, common_commands, project_summary,
                 tool_stats, updated_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(repo_path) DO UPDATE SET
                commit_style = CASE WHEN excluded.commit_style != ''
                    THEN excluded.commit_style ELSE repo_context.commit_style END,
                common_commands = CASE WHEN excluded.common_commands != ''
                    THEN excluded.common_commands ELSE repo_context.common_commands END,
                project_summary = CASE WHEN excluded.project_summary != ''
                    THEN excluded.project_summary ELSE repo_context.project_summary END,
                tool_stats = CASE WHEN excluded.tool_stats != ''
                    THEN excluded.tool_stats ELSE repo_context.tool_stats END,
                updated_at = excluded.updated_at
        """, (repo_path, commit_style, common_commands,
              project_summary, tool_stats))
        self._conn.commit()

    def get_repo_context(self, repo_path: str) -> dict | None:
        """Get accumulated context for a repo."""
        row = self._conn.execute(
            "SELECT * FROM repo_context WHERE repo_path = ?",
            (repo_path,),
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Skill Evolution — shadow learning version tracking

    def save_skill_version(self, skill_name: str, skill_json: str,
                           change_reason: str = "", claude_example: str = "",
                           local_example: str = "", session_id: str = "") -> int:
        """Save a skill snapshot before evolution. Returns the new version number."""
        # Get current max version for this skill
        row = self._conn.execute(
            "SELECT MAX(version) FROM skill_versions WHERE skill_name = ?",
            (skill_name,),
        ).fetchone()
        version = (row[0] or 0) + 1
        cur = self._conn.execute("""
            INSERT INTO skill_versions
                (skill_name, version, skill_json, change_reason,
                 claude_example, local_example, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (skill_name, version, skill_json, change_reason,
              claude_example, local_example, session_id))
        self._conn.commit()
        return version

    def get_skill_versions(self, skill_name: str, limit: int = 10) -> list[dict]:
        """Get version history for a skill, newest first."""
        rows = self._conn.execute("""
            SELECT * FROM skill_versions
            WHERE skill_name = ?
            ORDER BY version DESC LIMIT ?
        """, (skill_name, limit)).fetchall()
        return [dict(r) for r in rows]

    def get_skill_version(self, skill_name: str, version: int) -> dict | None:
        """Get a specific version of a skill."""
        row = self._conn.execute("""
            SELECT * FROM skill_versions
            WHERE skill_name = ? AND version = ?
        """, (skill_name, version)).fetchone()
        return dict(row) if row else None

    def get_latest_skill_version(self, skill_name: str) -> int:
        """Get the latest version number for a skill (0 if never versioned)."""
        row = self._conn.execute(
            "SELECT MAX(version) FROM skill_versions WHERE skill_name = ?",
            (skill_name,),
        ).fetchone()
        return row[0] or 0

    def get_evolved_skills_summary(self, limit: int = 20) -> list[dict]:
        """Get summary of all skill evolutions, grouped by skill."""
        rows = self._conn.execute("""
            SELECT skill_name, MAX(version) as latest_version,
                   COUNT(*) as total_versions,
                   MAX(created_at) as last_evolved
            FROM skill_versions
            GROUP BY skill_name
            ORDER BY last_evolved DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]

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
    # Response cache

    def cache_response(self, query: str, query_vector: list[float],
                       response: str, session_id: str = "") -> int:
        """Store a Claude Q→A pair for future re-use."""
        cur = self._conn.execute("""
            INSERT INTO response_cache
                (query, query_vector, response, session_id)
            VALUES (?, ?, ?, ?)
        """, (query[:2000], json.dumps(query_vector), response[:8000], session_id))
        self._conn.commit()
        return cur.lastrowid or 0

    def search_response_cache(self, query_vector: list[float],
                               min_sim: float = 0.88,
                               top_k: int = 3) -> list[dict]:
        """Find cached responses for semantically similar queries."""
        rows = self._conn.execute(
            "SELECT id, query, query_vector, response, hit_count, quality, created_at "
            "FROM response_cache"
        ).fetchall()
        qnorm = math.sqrt(sum(x * x for x in query_vector))
        if not qnorm:
            return []
        scored: list[tuple[float, dict]] = []
        for row in rows:
            vec = json.loads(row["query_vector"])
            snorm = math.sqrt(sum(x * x for x in vec))
            if not snorm:
                continue
            sim = sum(a * b for a, b in zip(query_vector, vec)) / (qnorm * snorm)
            if sim >= min_sim:
                d = {k: row[k] for k in ("id", "query", "response", "hit_count",
                                          "quality", "created_at")}
                d["similarity"] = sim
                scored.append((sim * float(row["quality"] or 1.0), d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]]

    def hit_response_cache(self, cache_id: int) -> None:
        """Increment hit count and update last_used_at."""
        self._conn.execute("""
            UPDATE response_cache
            SET hit_count = hit_count + 1, last_used_at = datetime('now')
            WHERE id = ?
        """, (cache_id,))
        self._conn.commit()

    def invalidate_response_cache(self, cache_id: int) -> None:
        """Mark a cached response as low quality (stale)."""
        self._conn.execute(
            "UPDATE response_cache SET quality = 0.1 WHERE id = ?", (cache_id,)
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Error cache

    def cache_error(self, error_text: str, error_vector: list[float],
                    fix_hint: str, session_id: str = "") -> int:
        cur = self._conn.execute("""
            INSERT INTO error_cache (error_text, error_vector, fix_hint, session_id)
            VALUES (?, ?, ?, ?)
        """, (error_text[:1000], json.dumps(error_vector), fix_hint[:2000], session_id))
        self._conn.commit()
        return cur.lastrowid or 0

    def search_error_cache(self, error_vector: list[float],
                            min_sim: float = 0.82) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, error_text, error_vector, fix_hint, confirmed, hit_count "
            "FROM error_cache"
        ).fetchall()
        qnorm = math.sqrt(sum(x * x for x in error_vector))
        if not qnorm:
            return []
        scored: list[tuple[float, dict]] = []
        for row in rows:
            vec = json.loads(row["error_vector"])
            snorm = math.sqrt(sum(x * x for x in vec))
            if not snorm:
                continue
            sim = sum(a * b for a, b in zip(error_vector, vec)) / (qnorm * snorm)
            if sim >= min_sim:
                d = {k: row[k] for k in ("id", "error_text", "fix_hint",
                                          "confirmed", "hit_count")}
                d["similarity"] = sim
                scored.append((sim, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:3]]

    def confirm_error_fix(self, cache_id: int) -> None:
        self._conn.execute("""
            UPDATE error_cache
            SET confirmed = 1, hit_count = hit_count + 1
            WHERE id = ?
        """, (cache_id,))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Message pattern tracking (for auto-skill generation)

    def record_message_pattern(self, message: str, vector: list[float],
                                min_sim: float = 0.85) -> dict:
        """Track recurring message patterns. Returns the pattern row + whether
        it crossed the repetition threshold for auto-skill generation."""
        rows = self._conn.execute(
            "SELECT id, canonical, vector, count, skill_generated FROM message_patterns"
        ).fetchall()
        qnorm = math.sqrt(sum(x * x for x in vector))
        if not qnorm:
            return {}

        best_id: int | None = None
        best_sim = 0.0
        for row in rows:
            vec = json.loads(row["vector"])
            snorm = math.sqrt(sum(x * x for x in vec))
            if not snorm:
                continue
            sim = sum(a * b for a, b in zip(vector, vec)) / (qnorm * snorm)
            if sim > best_sim:
                best_sim = sim
                if sim >= min_sim:
                    best_id = row["id"]

        if best_id is not None:
            self._conn.execute("""
                UPDATE message_patterns
                SET count = count + 1, last_seen_at = datetime('now')
                WHERE id = ?
            """, (best_id,))
            self._conn.commit()
            row_updated = self._conn.execute(
                "SELECT * FROM message_patterns WHERE id = ?", (best_id,)
            ).fetchone()
            return dict(row_updated)
        else:
            cur = self._conn.execute("""
                INSERT INTO message_patterns (canonical, vector, count)
                VALUES (?, ?, 1)
            """, (message[:500], json.dumps(vector)))
            self._conn.commit()
            return {"id": cur.lastrowid, "canonical": message, "count": 1,
                    "skill_generated": 0}

    def mark_pattern_skill_generated(self, pattern_id: int) -> None:
        self._conn.execute(
            "UPDATE message_patterns SET skill_generated = 1 WHERE id = ?",
            (pattern_id,)
        )
        self._conn.commit()

    def get_top_patterns(self, min_count: int = 3,
                         limit: int = 10) -> list[sqlite3.Row]:
        """Return patterns that recur often but haven't generated a skill yet."""
        return self._conn.execute("""
            SELECT id, canonical, count, skill_generated, last_seen_at
            FROM message_patterns
            WHERE count >= ? AND skill_generated = 0
            ORDER BY count DESC
            LIMIT ?
        """, (min_count, limit)).fetchall()

    # ------------------------------------------------------------------
    # Prompt patterns

    def save_prompt_pattern(self, trigger: str, trigger_vector: list[float],
                             pattern: str, context_type: str = "") -> int:
        """Store a successful prompt enrichment pattern."""
        # Check if very similar trigger already exists → increment instead
        rows = self._conn.execute(
            "SELECT id, trigger_vector FROM prompt_patterns"
        ).fetchall()
        qnorm = math.sqrt(sum(x * x for x in trigger_vector))
        for row in rows:
            vec = json.loads(row["trigger_vector"])
            snorm = math.sqrt(sum(x * x for x in vec))
            if snorm and qnorm:
                sim = sum(a * b for a, b in zip(trigger_vector, vec)) / (qnorm * snorm)
                if sim >= 0.90:
                    self._conn.execute(
                        "UPDATE prompt_patterns SET success_count = success_count + 1 WHERE id = ?",
                        (row["id"],)
                    )
                    self._conn.commit()
                    return row["id"]
        cur = self._conn.execute("""
            INSERT INTO prompt_patterns
                (trigger_text, trigger_vector, pattern_text, context_type)
            VALUES (?, ?, ?, ?)
        """, (trigger[:500], json.dumps(trigger_vector), pattern[:2000], context_type))
        self._conn.commit()
        return cur.lastrowid or 0

    def search_prompt_patterns(self, trigger_vector: list[float],
                                min_sim: float = 0.75) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, trigger_text, trigger_vector, pattern_text, "
            "context_type, success_count FROM prompt_patterns"
        ).fetchall()
        qnorm = math.sqrt(sum(x * x for x in trigger_vector))
        if not qnorm:
            return []
        scored: list[tuple[float, dict]] = []
        for row in rows:
            vec = json.loads(row["trigger_vector"])
            snorm = math.sqrt(sum(x * x for x in vec))
            if not snorm:
                continue
            sim = sum(a * b for a, b in zip(trigger_vector, vec)) / (qnorm * snorm)
            if sim >= min_sim:
                d = {k: row[k] for k in ("id", "trigger_text", "pattern_text",
                                          "context_type", "success_count")}
                d["similarity"] = sim
                scored.append((sim * row["success_count"], d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:3]]

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

    # ------------------------------------------------------------------
    # Background job queue — Phase A.2

    def enqueue_job(self, kind: str, payload: dict, priority: int = 5) -> int:
        """Enqueue a background job. Returns the job id."""
        cur = self._conn.execute(
            "INSERT INTO background_jobs (kind, payload, priority) VALUES (?, ?, ?)",
            (kind, json.dumps(payload), priority),
        )
        self._conn.commit()
        return cur.lastrowid or 0

    def dequeue_job(self) -> sqlite3.Row | None:
        """Atomically claim the highest-priority pending job. Returns None if queue empty."""
        # Single statement: claim + return in one atomic write
        row = self._conn.execute(
            """
            UPDATE background_jobs
            SET status     = 'running',
                started_at = datetime('now'),
                attempts   = attempts + 1
            WHERE id = (
                SELECT id FROM background_jobs
                WHERE status = 'pending'
                ORDER BY priority ASC, created_at ASC
                LIMIT 1
            )
            RETURNING *
            """
        ).fetchone()
        if row is not None:
            self._conn.commit()
        return row

    def complete_job(self, job_id: int, result: dict | None = None, worker: str = "") -> None:
        self._conn.execute(
            "UPDATE background_jobs SET status = 'done', completed_at = datetime('now'),"
            " result = ?, worker_used = ? WHERE id = ?",
            (json.dumps(result or {}), worker, job_id),
        )
        self._conn.commit()

    def fail_job(self, job_id: int, error: str, max_attempts: int = 3) -> None:
        """Fail or defer a job based on attempt count."""
        self._conn.execute(
            """
            UPDATE background_jobs
            SET attempts = attempts + 1,
                status   = CASE WHEN attempts + 1 >= ? THEN 'failed' ELSE 'deferred' END,
                error    = ?
            WHERE id = ?
            """,
            (max_attempts, error[:1000], job_id),
        )
        self._conn.commit()

    def pending_job_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM background_jobs WHERE status IN ('pending','deferred')"
        ).fetchone()
        return row[0] if row else 0

    def reset_deferred_jobs(self) -> int:
        """Move deferred jobs back to pending so they can be retried."""
        cur = self._conn.execute(
            "UPDATE background_jobs SET status = 'pending' WHERE status = 'deferred'"
        )
        self._conn.commit()
        return cur.rowcount

    # ------------------------------------------------------------------
    # Cron jobs — Phase H.1

    def _seed_builtin_cron_jobs(self) -> None:
        """Seed built-in cron jobs on first run (idempotent due to ON CONFLICT)."""
        defaults = [
            ("memory-optimize-preview", "Memory optimize (dry-run preview)", "0 2 * * *",
             "optimize_memory_dry_run", {}, True, False),
            ("teachings-sync", "Teachings sync from feedback files", "0 3 * * *",
             "feedback_to_teachings", {}, True, False),
            ("archive-closed-tasks", "Archive DONE/SHIPPED entries to DB", "0 4 * * *",
             "archive_memory_to_db_dry_run", {}, True, True),
            ("memory-export-snapshot", "Weekly memory export snapshot", "0 0 * * 0",
             "memexp_snapshot_create", {}, False, False),
            ("pipeline-health-check", "Pipeline backend health check", "*/15 * * * *",
             "check_embedding_backends", {}, True, False),
            ("codegraph-sync", "Keep configured CodeGraph indexes fresh (incremental)",
             "*/30 * * * *", "codegraph_sync", {}, True, False),
            ("wiki-reindex-nightly", "Wiki vault nightly reindex + re-embed", "0 5 * * *",
             "wiki_reindex_nightly", {}, False, False),
            ("discussions-sync-nightly", "GitHub Discussions periodic sync into wiki", "0 1 * * *",
             "discussions_sync_nightly", {}, False, False),
        ]
        for name, desc, sched, cmd, params, enabled, is_dangerous in defaults:
            self._conn.execute(
                """INSERT INTO cron_jobs (name, description, schedule, command, params,
                       enabled, is_builtin, is_dangerous)
                   VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                   ON CONFLICT(name) DO UPDATE SET
                       description=excluded.description,
                       schedule=excluded.schedule,
                       command=excluded.command,
                       is_builtin=1,
                       updated_at=datetime('now')""",
                (name, desc, sched, cmd, json.dumps(params),
                 1 if enabled else 0, 1 if is_dangerous else 0),
            )
        self._conn.commit()

    def list_cron_jobs(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM cron_jobs ORDER BY is_builtin DESC, name ASC"
        ).fetchall()

    def get_cron_job(self, job_id: int) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM cron_jobs WHERE id = ?", (job_id,)
        ).fetchone()

    def upsert_cron_job(self, name: str, description: str, schedule: str,
                        command: str, params: dict, enabled: bool = True,
                        is_builtin: bool = False, is_dangerous: bool = False) -> int:
        cur = self._conn.execute(
            """INSERT INTO cron_jobs (name, description, schedule, command, params,
                   enabled, is_builtin, is_dangerous)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(name) DO UPDATE SET
                   description=excluded.description,
                   schedule=excluded.schedule,
                   command=excluded.command,
                   params=excluded.params,
                   enabled=excluded.enabled,
                   is_dangerous=excluded.is_dangerous,
                   updated_at=datetime('now')""",
            (name, description, schedule, command, json.dumps(params),
             1 if enabled else 0, 1 if is_builtin else 0, 1 if is_dangerous else 0),
        )
        self._conn.commit()
        return cur.lastrowid or 0

    def delete_cron_job(self, job_id: int) -> bool:
        """Delete non-builtin cron job."""
        cur = self._conn.execute(
            "DELETE FROM cron_jobs WHERE id = ? AND is_builtin = 0", (job_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def update_cron_job_status(self, job_id: int, status: str,
                               duration_ms: int | None = None,
                               error: str | None = None) -> None:
        self._conn.execute(
            """UPDATE cron_jobs SET last_run_at=datetime('now'), last_status=?,
                   last_duration_ms=?, last_error=?, updated_at=datetime('now')
               WHERE id=?""",
            (status, duration_ms, error, job_id),
        )
        self._conn.commit()

    def reset_cron_job_for_run(self, job_id: int) -> None:
        """Mark a cron job as pending for immediate run (resets last_run_at)."""
        self._conn.execute(
            """UPDATE cron_jobs
               SET last_run_at='2000-01-01T00:00:00', last_status='pending',
                   updated_at=datetime('now')
               WHERE id=?""",
            (job_id,),
        )
        self._conn.commit()

    def toggle_cron_job(self, job_id: int, enabled: bool) -> bool:
        cur = self._conn.execute(
            "UPDATE cron_jobs SET enabled=?, updated_at=datetime('now') WHERE id=?",
            (1 if enabled else 0, job_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def update_cron_job(self, job_id: int, **fields: object) -> bool:
        """Update one or more fields of a cron job by id.

        Accepted keyword arguments: schedule, command, enabled, description.
        Unknown keys are silently ignored. Returns True if a row was updated.
        """
        allowed = {"schedule", "command", "enabled", "description"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return False
        set_clause = ", ".join(f"{col}=?" for col in updates)
        values = list(updates.values()) + [job_id]
        cur = self._conn.execute(
            f"UPDATE cron_jobs SET {set_clause}, updated_at=datetime('now') WHERE id=?",
            values,
        )
        self._conn.commit()
        return cur.rowcount > 0

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Plugin extension-point: A7 — plugin-scoped DB access
    # See docs/plugin-extension-points.md
    # ------------------------------------------------------------------

    def plugin_db(self, namespace: str) -> "PluginStore":
        """Return a namespace-guarded wrapper around the shared sqlite connection.

        On first call per-process, if the plugin ships ``storage.schema``
        (an SQL file containing only ``CREATE TABLE IF NOT EXISTS
        plugin_{namespace}_*`` statements) the schema is applied and its
        sha256 recorded in ``plugin_migrations``. Subsequent calls are no-ops
        unless the schema file's hash changes.
        """
        if not namespace or not namespace.replace("_", "").isalnum():
            raise ValueError(f"invalid plugin namespace: {namespace!r}")
        ps = PluginStore(self._conn, namespace)
        _maybe_apply_plugin_schema(self._conn, namespace)
        return ps

    # ------------------------------------------------------------------
    # Plugin extension-point: A8 — namespaced vector index
    # ------------------------------------------------------------------

    def upsert_vector(self, namespace: str, doc_id: str, text: str,
                      metadata: dict | None = None,
                      model: str | None = None,
                      level: str | None = None,
                      source: str | None = None,
                      project: str | None = None,
                      tags: list[str] | None = None) -> None:
        """Embed ``text`` and upsert into the shared ``vectors`` table.

        Phase M1: ``level`` defaults to the namespace's ``default_level`` from
        ``vector_index_config`` (falling back to L2). ``source``/``project``/
        ``tags`` are opaque provenance fields used by retrieval profiles.
        """
        from .embeddings import embed as _embed, EMBED_MODEL
        m = model or EMBED_MODEL
        vec = _embed(text, model=m)
        norm = math.sqrt(sum(x * x for x in vec))
        meta_json = json.dumps(metadata) if metadata is not None else None
        tags_json = json.dumps(tags) if tags else None
        if level is None:
            row = self._conn.execute(
                "SELECT default_level FROM vector_index_config WHERE name = ?",
                (namespace,),
            ).fetchone()
            level = (row["default_level"] if row else None) or "L2"
        self._conn.execute(
            """
            INSERT INTO vectors (namespace, doc_id, model, vector, norm, metadata,
                                 level, source, project, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(namespace, doc_id) DO UPDATE SET
                model      = excluded.model,
                vector     = excluded.vector,
                norm       = excluded.norm,
                metadata   = excluded.metadata,
                level      = excluded.level,
                source     = COALESCE(excluded.source,  vectors.source),
                project    = COALESCE(excluded.project, vectors.project),
                tags       = COALESCE(excluded.tags,    vectors.tags),
                indexed_at = datetime('now')
            """,
            (namespace, doc_id, m, json.dumps(vec), norm, meta_json,
             level, source, project, tags_json),
        )
        self._conn.commit()

    def search_vectors(self, query: str,
                       namespaces: list[str] | None = None,
                       top_k: int = 5,
                       similarity_threshold: float = 0.0,
                       levels: list[str] | None = None,
                       apply_level_weight: bool = True,
                       apply_recency_decay: bool = True) -> list[dict]:
        """Cosine-search the namespaced ``vectors`` table with level weighting.

        Phase M1: multiplies raw cosine similarity by the entry's level weight
        and an exponential recency-decay factor (``exp(-age_days / half_life)``).
        ``levels=None`` keeps all levels; pass e.g. ``["L3","L4"]`` to drop
        ephemeral chatter. Returns ``{namespace, doc_id, score, raw_score,
        level, metadata}`` sorted by weighted score desc.
        """
        from .embeddings import embed as _embed
        import time as _time
        from datetime import datetime as _dt

        qvec = _embed(query)
        qnorm = math.sqrt(sum(x * x for x in qvec))
        if qnorm == 0.0:
            return []

        where = []
        params: list = []
        if namespaces:
            where.append(f"namespace IN ({','.join('?' * len(namespaces))})")
            params.extend(namespaces)
        if levels:
            where.append(f"level IN ({','.join('?' * len(levels))})")
            params.extend(levels)
        sql = ("SELECT namespace, doc_id, vector, norm, metadata, level, indexed_at "
               "FROM vectors")
        if where:
            sql += " WHERE " + " AND ".join(where)
        rows = self._conn.execute(sql, tuple(params)).fetchall()

        # Cache per-namespace half-life to avoid N queries.
        half_life_cache: dict[str, float] = {}
        def _half_life(ns: str, lvl: str) -> float:
            key = f"{ns}|{lvl}"
            if key in half_life_cache:
                return half_life_cache[key]
            row = self._conn.execute(
                "SELECT half_life_days FROM vector_index_config WHERE name = ?",
                (ns,),
            ).fetchone()
            hl = float(row["half_life_days"]) if row else LEVEL_HALF_LIFE_DAYS.get(lvl, 30.0)
            half_life_cache[key] = hl
            return hl

        now_ts = _time.time()
        scored: list[tuple[float, dict]] = []
        for row in rows:
            snorm = row["norm"] or 0.0
            if snorm == 0.0:
                continue
            vec = json.loads(row["vector"])
            dot = sum(a * b for a, b in zip(qvec, vec))
            raw = dot / (qnorm * snorm)
            if raw < similarity_threshold:
                continue
            lvl = row["level"] or "L2"
            weight = LEVEL_WEIGHTS.get(lvl, 1.0) if apply_level_weight else 1.0
            decay = 1.0
            if apply_recency_decay and row["indexed_at"]:
                try:
                    ts = _dt.fromisoformat(row["indexed_at"]).timestamp()
                    age_days = max(0.0, (now_ts - ts) / 86400.0)
                    hl = _half_life(row["namespace"], lvl)
                    if hl > 0:
                        decay = math.exp(-age_days / hl)
                except (TypeError, ValueError):
                    pass
            score = raw * weight * decay
            meta = None
            if row["metadata"]:
                try:
                    meta = json.loads(row["metadata"])
                except (TypeError, ValueError):
                    meta = None
            scored.append((score, {
                "namespace": row["namespace"],
                "doc_id": row["doc_id"],
                "score": score,
                "raw_score": raw,
                "level": lvl,
                "metadata": meta,
            }))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    # ------------------------------------------------------------------
    # M4-3 federation-lite — cross-host read-only views
    # ------------------------------------------------------------------

    def federation_view(
        self,
        remote_db_path: str | Path,
        alias: str = "remote",
    ) -> dict:
        """Open a read-only ATTACH on another host's skill-hub DB and report
        what's visible (tasks + events counts, distinct ``node_id`` values).

        Intended for a shared/synced filesystem layout (Syncthing, rsync,
        git-annex). No network protocol — just a SQLite ATTACH against a
        peer's file. The remote DB is detached before returning so the local
        connection stays clean.

        Returns a dict with::

            {
              "alias":        attached schema alias used in queries,
              "remote_path":  resolved absolute path,
              "local_node":   self.node_id,
              "remote_nodes": [<node_id>, ...],   # distinct in remote.tasks ∪ events
              "tasks":        {"local": N, "remote": N},
              "events":       {"local": N, "remote": N},
              "schemas":      {"events_remote": bool, "tasks_remote": bool},
            }
        """
        path = Path(remote_db_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"remote skill-hub DB not found: {path}")

        safe_alias = re.sub(r"[^A-Za-z0-9_]+", "_", alias).strip("_") or "remote"
        # SQLite ATTACH supports a 'file:' URI with read-only mode.
        uri = f"file:{path}?mode=ro&immutable=0"
        try:
            self._conn.execute(
                f"ATTACH DATABASE ? AS {safe_alias}", (uri,)
            )
        except sqlite3.OperationalError:
            # Older SQLite builds may reject the URI form for ATTACH; fall
            # back to a plain path (still respects WAL concurrency).
            self._conn.execute(
                f"ATTACH DATABASE ? AS {safe_alias}", (str(path),)
            )

        try:
            # Probe schemas — a non-skill-hub DB at this path should still
            # report cleanly instead of raising deep inside the caller.
            tbls = {
                row["name"]
                for row in self._conn.execute(
                    f"SELECT name FROM {safe_alias}.sqlite_master "
                    "WHERE type='table'"
                ).fetchall()
            }
            has_tasks = "tasks" in tbls
            has_events = "events" in tbls

            local_tasks = self._conn.execute(
                "SELECT COUNT(*) FROM tasks"
            ).fetchone()[0]
            local_events = self._conn.execute(
                "SELECT COUNT(*) FROM events"
            ).fetchone()[0]
            remote_tasks = (
                self._conn.execute(
                    f"SELECT COUNT(*) FROM {safe_alias}.tasks"
                ).fetchone()[0]
                if has_tasks
                else 0
            )
            remote_events = (
                self._conn.execute(
                    f"SELECT COUNT(*) FROM {safe_alias}.events"
                ).fetchone()[0]
                if has_events
                else 0
            )

            nodes: set[str] = set()
            if has_tasks:
                for row in self._conn.execute(
                    f"SELECT DISTINCT node_id FROM {safe_alias}.tasks "
                    "WHERE node_id IS NOT NULL"
                ).fetchall():
                    nodes.add(row[0])
            if has_events:
                for row in self._conn.execute(
                    f"SELECT DISTINCT node_id FROM {safe_alias}.events "
                    "WHERE node_id IS NOT NULL"
                ).fetchall():
                    nodes.add(row[0])

            return {
                "alias":        safe_alias,
                "remote_path":  str(path),
                "local_node":   self.node_id,
                "remote_nodes": sorted(nodes),
                "tasks":        {"local": local_tasks, "remote": remote_tasks},
                "events":       {"local": local_events, "remote": remote_events},
                "schemas":      {"events_remote": has_events, "tasks_remote": has_tasks},
            }
        finally:
            try:
                self._conn.execute(f"DETACH DATABASE {safe_alias}")
            except sqlite3.OperationalError:
                pass

    def append_event(
        self,
        session_id: str,
        kind: str,
        payload: dict | str,
        tool_name: str | None = None,
        ts: float | None = None,
        source: str | None = None,
    ) -> int | None:
        """Append a row to the M2 W1 event log.

        Exception-safe: any failure is logged at DEBUG level and returns None
        so a failed event append never breaks a tool call.  Returns the new
        row id on success.

        ``source`` overrides the ``node_id`` column; defaults to
        ``self.node_id``.  Payload is JSON-encoded (best-effort: large values
        are truncated, un-serialisable objects fall back to str()).
        """
        import time

        try:
            if ts is None:
                ts = time.time()
            node = source if source is not None else self.node_id
            if not isinstance(payload, str):
                try:
                    serialised = json.dumps(payload, default=str)
                except Exception:  # noqa: BLE001
                    serialised = json.dumps({"_raw": str(payload)[:1000]})
            else:
                serialised = payload
            # Cap payload at 64 KB so a single runaway event can't bloat the DB.
            if len(serialised) > 65536:
                serialised = serialised[:65536]
            cur = self._conn.execute(
                "INSERT INTO events (session_id, ts, kind, tool_name, payload, node_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, ts, kind, tool_name, serialised, node),
            )
            self._conn.commit()
            return int(cur.lastrowid or 0)
        except Exception as exc:  # noqa: BLE001
            _log.debug("append_event failed (non-fatal): %s", exc)
            return None

    def get_events(
        self,
        session_id: str = "",
        since: float = 0.0,
        kind: str = "",
        limit: int = 200,
    ) -> list[dict]:
        """Query the event log with optional filters.

        All filters are ANDed.  Empty ``session_id`` returns events across
        all sessions (still bounded by ``limit``).  Results are ordered by
        ``ts`` ascending.
        """
        clauses: list[str] = []
        params: list = []
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if since > 0.0:
            clauses.append("ts >= ?")
            params.append(since)
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        safe_limit = max(1, min(int(limit), 10000))
        params.append(safe_limit)
        try:
            rows = self._conn.execute(
                f"SELECT id, session_id, ts, kind, tool_name, payload, node_id "
                f"FROM events {where} ORDER BY ts ASC LIMIT ?",
                params,
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:  # noqa: BLE001
            _log.debug("get_events failed: %s", exc)
            return []

    def get_compression_stats(self, limit: int = 5000) -> dict:
        """Aggregate ``kind='compression'`` events into a UI/report summary.

        Each compression event payload carries ``bytes_before``/``bytes_after``/
        ``ratio``/``strategy``/``lossy``/``site``.  We aggregate over the most
        recent ``limit`` events (bounded scan; events are prunable).  Returns a
        dict that ``token_stats`` and the dashboard health card both render::

            {
              "calls": int,            # attempts that reached the compressor
              "hits": int,             # attempts that actually shrank the payload
              "bytes_before": int, "bytes_after": int, "saved": int,
              "avg_ratio": float,      # mean ratio over hits (1.0 = no change)
              "tokens_saved": int,     # saved // ~4 chars-per-token
              "by_strategy": {strat: {"count": int, "saved": int}},
              "by_site": {site: {"count": int, "saved": int}},
            }
        """
        empty = {
            "calls": 0, "hits": 0, "bytes_before": 0, "bytes_after": 0,
            "saved": 0, "avg_ratio": 1.0, "tokens_saved": 0,
            "by_strategy": {}, "by_site": {},
        }
        try:
            rows = self._conn.execute(
                "SELECT payload FROM events WHERE kind='compression' "
                "ORDER BY ts DESC LIMIT ?",
                (max(1, min(int(limit), 50000)),),
            ).fetchall()
        except Exception as exc:  # noqa: BLE001
            _log.debug("get_compression_stats query failed: %s", exc)
            return empty
        if not rows:
            return empty

        calls = hits = b_before = b_after = 0
        ratio_sum = 0.0
        by_strategy: dict[str, dict] = {}
        by_site: dict[str, dict] = {}
        for r in rows:
            try:
                p = json.loads(r["payload"])
            except Exception:  # noqa: BLE001
                continue
            bb = int(p.get("bytes_before") or 0)
            ba = int(p.get("bytes_after") or 0)
            strat = str(p.get("strategy") or "?")
            site = str(p.get("site") or "?")
            calls += 1
            b_before += bb
            b_after += ba
            saved = max(0, bb - ba)
            if saved > 0:
                hits += 1
                ratio_sum += (ba / bb) if bb else 1.0
            sd = by_strategy.setdefault(strat, {"count": 0, "saved": 0})
            sd["count"] += 1
            sd["saved"] += saved
            td = by_site.setdefault(site, {"count": 0, "saved": 0})
            td["count"] += 1
            td["saved"] += saved

        saved = max(0, b_before - b_after)
        return {
            "calls": calls,
            "hits": hits,
            "bytes_before": b_before,
            "bytes_after": b_after,
            "saved": saved,
            "avg_ratio": (ratio_sum / hits) if hits else 1.0,
            "tokens_saved": saved // 4,
            "by_strategy": by_strategy,
            "by_site": by_site,
        }

    def get_llm_stats(self, limit: int = 5000) -> dict:
        """Aggregate ``kind='llm_call'`` events into a UI/report summary.

        Each llm_call event payload carries ``op``/``model``/``tier``/
        ``duration_ms``/``prompt_tokens``/``completion_tokens``/
        ``total_tokens``/``status``.  Aggregates over the most recent
        ``limit`` events.  Returns a dict that ``token_stats`` and the
        dashboard health card both render::

            {
              "calls": int, "errors": int,
              "total_duration_ms": int, "avg_latency_ms": float,
              "prompt_tokens": int, "completion_tokens": int, "total_tokens": int,
              "tokens_per_sec": float,
              "by_op": {op: {"count": int, "total_tokens": int, "duration_ms": int}},
              "by_model": {model: {"count": int, "errors": int,
                                   "total_tokens": int, "duration_ms": int}},
            }
        """
        empty: dict = {
            "calls": 0, "errors": 0,
            "total_duration_ms": 0, "avg_latency_ms": 0.0,
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "tokens_per_sec": 0.0,
            "by_op": {}, "by_model": {},
        }
        try:
            rows = self._conn.execute(
                "SELECT payload FROM events WHERE kind='llm_call' "
                "ORDER BY ts DESC LIMIT ?",
                (max(1, min(int(limit), 50000)),),
            ).fetchall()
        except Exception as exc:  # noqa: BLE001
            _log.debug("get_llm_stats query failed: %s", exc)
            return empty
        if not rows:
            return empty

        calls = errors = 0
        total_duration_ms = 0
        ok_duration_ms = 0
        prompt_tokens = completion_tokens = total_tokens = 0
        by_op: dict[str, dict] = {}
        by_model: dict[str, dict] = {}
        for r in rows:
            try:
                p = json.loads(r["payload"])
            except Exception:  # noqa: BLE001
                continue
            dur = int(p.get("duration_ms") or 0)
            pt = int(p.get("prompt_tokens") or 0)
            ct = int(p.get("completion_tokens") or 0)
            tt = int(p.get("total_tokens") or 0)
            op = str(p.get("op") or "?")
            model = str(p.get("model") or "?")
            status = str(p.get("status") or "ok")
            calls += 1
            total_duration_ms += dur
            if status != "ok":
                errors += 1
            else:
                ok_duration_ms += dur
            prompt_tokens += pt
            completion_tokens += ct
            total_tokens += tt
            od = by_op.setdefault(op, {"count": 0, "total_tokens": 0, "duration_ms": 0})
            od["count"] += 1
            od["total_tokens"] += tt
            od["duration_ms"] += dur
            md = by_model.setdefault(
                model, {"count": 0, "errors": 0, "total_tokens": 0, "duration_ms": 0})
            md["count"] += 1
            if status != "ok":
                md["errors"] += 1
            md["total_tokens"] += tt
            md["duration_ms"] += dur

        avg_latency_ms = (total_duration_ms / calls) if calls else 0.0
        ok_duration_s = ok_duration_ms / 1000.0
        tokens_per_sec = (completion_tokens / ok_duration_s) if ok_duration_s > 0 else 0.0
        return {
            "calls": calls,
            "errors": errors,
            "total_duration_ms": total_duration_ms,
            "avg_latency_ms": round(avg_latency_ms, 1),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "tokens_per_sec": round(tokens_per_sec, 1),
            "by_op": by_op,
            "by_model": by_model,
        }

    def events_prune(
        self,
        before_ts: float = 0.0,
        dry_run: bool = False,
        retention_days: int | None = None,
    ) -> dict:
        """Prune closed sessions whose events are older than the retention window.

        Rules:
        * ``retention_days`` defaults to the config value ``event_log_retention_days``
          (default 30).  ``0`` = keep forever (no-op).
        * ``before_ts`` overrides the config window when > 0.
        * A session is "closed" iff it has at least one ``session_end`` event.
        * Sessions whose only rows are already ``session_snapshot`` kind are
          skipped (re-run guard).
        * Closed sessions whose latest event ``ts`` < cutoff: coalesce all raw
          events into one ``session_snapshot`` row then delete the originals.
        * In-flight sessions (no ``session_end``) are never pruned.
        * ``dry_run=True``: report candidates + counts, delete nothing.

        Returns a summary dict with keys ``candidates``, ``rows_deleted``,
        ``snapshots_written``, ``dry_run``.
        """
        import time

        try:
            from . import config as _cfg
            _ret = _cfg.get("event_log_retention_days")
            cfg_days = int(_ret) if _ret is not None else 30
        except Exception:  # noqa: BLE001
            cfg_days = 30
        if retention_days is not None:
            cfg_days = retention_days

        if cfg_days == 0:
            return {"candidates": 0, "rows_deleted": 0, "snapshots_written": 0, "dry_run": dry_run}

        now = time.time()
        cutoff = before_ts if before_ts > 0.0 else (now - cfg_days * 86400.0)

        try:
            # Sessions that have a session_end event.
            closed_rows = self._conn.execute(
                "SELECT DISTINCT session_id FROM events WHERE kind = 'session_end'"
            ).fetchall()
            closed_sessions = {r[0] for r in closed_rows}
        except Exception as exc:  # noqa: BLE001
            _log.debug("events_prune: closed session query failed: %s", exc)
            return {"candidates": 0, "rows_deleted": 0, "snapshots_written": 0, "dry_run": dry_run, "error": str(exc)}

        candidates: list[str] = []
        for sid in closed_sessions:
            try:
                # Skip sessions that are already fully coalesced (only snapshots).
                kinds = self._conn.execute(
                    "SELECT DISTINCT kind FROM events WHERE session_id = ?", (sid,)
                ).fetchall()
                kind_set = {r[0] for r in kinds}
                if kind_set and kind_set.issubset({"session_snapshot"}):
                    continue

                # Check if the latest event for this session is older than cutoff.
                latest_row = self._conn.execute(
                    "SELECT MAX(ts) AS max_ts FROM events WHERE session_id = ?", (sid,)
                ).fetchone()
                if not latest_row or latest_row[0] is None or latest_row[0] >= cutoff:
                    continue
                candidates.append(sid)
            except Exception as exc:  # noqa: BLE001
                _log.debug("events_prune: per-session check failed for %s: %s", sid, exc)

        rows_deleted = 0
        snapshots_written = 0

        for sid in candidates:
            try:
                raw_rows = self._conn.execute(
                    "SELECT id, ts, kind, tool_name, payload FROM events WHERE session_id = ?",
                    (sid,),
                ).fetchall()
                if not raw_rows:
                    continue

                # Build compact summary.
                kind_counts: dict[str, int] = {}
                tool_tallies: dict[str, int] = {}
                ts_values: list[float] = []
                for r in raw_rows:
                    kind_counts[r["kind"]] = kind_counts.get(r["kind"], 0) + 1
                    if r["tool_name"]:
                        tool_tallies[r["tool_name"]] = tool_tallies.get(r["tool_name"], 0) + 1
                    if r["ts"] is not None:
                        ts_values.append(float(r["ts"]))

                snapshot_payload = json.dumps({
                    "kind_counts": kind_counts,
                    "tool_tallies": tool_tallies,
                    "first_ts": min(ts_values) if ts_values else None,
                    "last_ts": max(ts_values) if ts_values else None,
                    "raw_row_count": len(raw_rows),
                })
                snapshot_ts = max(ts_values) if ts_values else now

                if not dry_run:
                    self._conn.execute(
                        "DELETE FROM events WHERE session_id = ?", (sid,)
                    )
                    self._conn.execute(
                        "INSERT INTO events (session_id, ts, kind, tool_name, payload, node_id) "
                        "VALUES (?, ?, 'session_snapshot', NULL, ?, ?)",
                        (sid, snapshot_ts, snapshot_payload, self.node_id),
                    )
                    self._conn.commit()
                    snapshots_written += 1
                    rows_deleted += len(raw_rows)
                else:
                    rows_deleted += len(raw_rows)  # reported as "would delete"
            except Exception as exc:  # noqa: BLE001
                _log.debug("events_prune: coalesce failed for %s: %s", sid, exc)

        return {
            "candidates": len(candidates),
            "rows_deleted": rows_deleted,
            "snapshots_written": snapshots_written if not dry_run else 0,
            "dry_run": dry_run,
        }


# ----------------------------------------------------------------------
# Plugin extension-point: A7 — PluginStore wrapper
# Enforces that writes touch only plugin_{namespace}_* tables. Reads on
# core tables are allowed so plugins can correlate to session_log/tasks.
# ----------------------------------------------------------------------

import re as _re  # local import to avoid shuffling file headers

_SQL_WRITE_RE = _re.compile(
    r"\b(CREATE|INSERT|UPDATE|DELETE|DROP|ALTER|REPLACE|TRUNCATE)\b",
    _re.IGNORECASE,
)
_SQL_TABLE_RE = _re.compile(
    r"\b(?:FROM|INTO|UPDATE|TABLE|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)",
    _re.IGNORECASE,
)


class PluginStore:
    """Thin namespace-guarded wrapper around the shared sqlite connection.

    Rejects DDL/DML touching any table whose name doesn't start with
    ``plugin_{namespace}_``. Bare ``SELECT`` may reference core tables
    (read-only correlation).
    """

    def __init__(self, conn: sqlite3.Connection, namespace: str) -> None:
        self._conn = conn
        self.namespace = namespace
        self._prefix = f"plugin_{namespace}_"

    def _guard(self, sql: str) -> None:
        upper = sql.strip().upper()
        is_write = bool(_SQL_WRITE_RE.search(upper))
        if not is_write:
            return  # read-only: any table is allowed
        tables = _SQL_TABLE_RE.findall(sql)
        for t in tables:
            if not t.lower().startswith(self._prefix):
                raise PermissionError(
                    f"PluginStore({self.namespace!r}): write to table "
                    f"{t!r} rejected — only {self._prefix}* tables allowed"
                )

    def execute(self, sql: str, params: tuple | list | dict = ()) -> sqlite3.Cursor:
        self._guard(sql)
        cur = self._conn.execute(sql, params)
        self._conn.commit()
        return cur

    def fetch_all(self, sql: str, params: tuple | list | dict = ()) -> list[sqlite3.Row]:
        self._guard(sql)
        return self._conn.execute(sql, params).fetchall()

    def fetch_one(self, sql: str, params: tuple | list | dict = ()) -> sqlite3.Row | None:
        self._guard(sql)
        return self._conn.execute(sql, params).fetchone()


# Idempotent per-process cache of "already bootstrapped" namespaces
_PLUGIN_SCHEMA_APPLIED: dict[str, str] = {}


def _maybe_apply_plugin_schema(conn: sqlite3.Connection, namespace: str) -> None:
    """Look up the plugin's ``storage.schema`` file and apply if changed.

    Schema file must only contain ``CREATE TABLE IF NOT EXISTS plugin_{namespace}_*``
    statements — any other statement causes the whole apply to abort.
    """
    import hashlib
    try:
        from .plugin_registry import iter_enabled_plugins
    except Exception:  # noqa: BLE001
        return

    for p in iter_enabled_plugins():
        storage = p["manifest"].get("storage") or {}
        if storage.get("namespace") != namespace:
            continue
        schema_rel = storage.get("schema")
        if not schema_rel:
            return
        schema_path = Path(p["path"]) / schema_rel
        if not schema_path.exists():
            return
        try:
            text = schema_path.read_text(encoding="utf-8")
        except OSError:
            return
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if _PLUGIN_SCHEMA_APPLIED.get(namespace) == sha:
            return
        row = conn.execute(
            "SELECT schema_hash FROM plugin_migrations WHERE namespace = ?",
            (namespace,),
        ).fetchone()
        if row is not None and row["schema_hash"] == sha:
            _PLUGIN_SCHEMA_APPLIED[namespace] = sha
            return

        # Validate: every non-empty, non-comment statement must match the
        # allowed pattern.
        prefix = f"plugin_{namespace}_"
        stmts = [s.strip() for s in text.split(";") if s.strip()]
        allow_re = _re.compile(
            r"^\s*CREATE\s+(?:UNIQUE\s+)?(?:TABLE|INDEX)\s+IF\s+NOT\s+EXISTS\s+"
            r"(?:[A-Za-z_][A-Za-z0-9_]*\s+ON\s+)?" + _re.escape(prefix),
            _re.IGNORECASE,
        )
        for stmt in stmts:
            # Strip leading comments
            cleaned = _re.sub(r"^\s*(--.*\n)+", "", stmt).strip()
            if not cleaned:
                continue
            if not allow_re.match(cleaned):
                # Log via stdlib; don't raise — keep plugin isolation soft.
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "plugin %s: schema.sql rejected — statement outside "
                    "plugin_%s_* table scope: %s",
                    namespace, namespace, cleaned[:80],
                )
                return
        try:
            with conn:
                conn.executescript(text)
                conn.execute(
                    "INSERT INTO plugin_migrations (namespace, schema_hash) "
                    "VALUES (?, ?) "
                    "ON CONFLICT(namespace) DO UPDATE SET "
                    "schema_hash = excluded.schema_hash, "
                    "applied_at = datetime('now')",
                    (namespace, sha),
                )
            _PLUGIN_SCHEMA_APPLIED[namespace] = sha
        except sqlite3.Error as exc:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "plugin %s: schema apply failed: %s", namespace, exc,
            )
        return

    # ------------------------------------------------------------------
    # Cron jobs CRUD
    # ------------------------------------------------------------------

    def list_cron_jobs(self) -> list[sqlite3.Row]:
        self._conn.row_factory = sqlite3.Row
        return self._conn.execute(
            "SELECT * FROM cron_jobs ORDER BY id"
        ).fetchall()

    def get_cron_job(self, job_id: int) -> sqlite3.Row | None:
        self._conn.row_factory = sqlite3.Row
        return self._conn.execute(
            "SELECT * FROM cron_jobs WHERE id=?", (job_id,)
        ).fetchone()

    def upsert_cron_job(
        self,
        name: str,
        schedule: str,
        command: str,
        enabled: bool = True,
        description: str = "",
        params: dict | None = None,
        is_builtin: bool = False,
        is_dangerous: bool = False,
    ) -> int:
        """Insert or replace a cron job; returns the row id."""
        cur = self._conn.execute(
            "INSERT INTO cron_jobs(name, schedule, command, enabled)"
            " VALUES(?,?,?,?)"
            " ON CONFLICT(name) DO UPDATE SET"
            " schedule=excluded.schedule, command=excluded.command,"
            " enabled=excluded.enabled",
            (name, schedule, command, int(enabled)),
        )
        self._conn.commit()
        if cur.lastrowid:
            return cur.lastrowid
        row = self._conn.execute(
            "SELECT id FROM cron_jobs WHERE name=?", (name,)
        ).fetchone()
        return row[0] if row else 0

    def toggle_cron_job(self, job_id: int, enabled: bool) -> None:
        self._conn.execute(
            "UPDATE cron_jobs SET enabled=? WHERE id=?",
            (int(enabled), job_id),
        )
        self._conn.commit()

    def delete_cron_job(self, job_id: int) -> bool:
        """Delete a non-builtin cron job. Returns True if deleted."""
        cur = self._conn.execute(
            "DELETE FROM cron_jobs WHERE id=? AND is_builtin=0", (job_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def update_cron_job_status(
        self,
        job_id: int,
        status: str,
        error: str | None = None,
        duration_ms: int | None = None,
    ) -> None:
        self._conn.execute(
            "UPDATE cron_jobs"
            " SET last_run_at=datetime('now'), last_status=?,"
            " last_error=?, last_duration_ms=?, run_count=run_count+1"
            " WHERE id=?",
            (status, error, duration_ms, job_id),
        )
        self._conn.commit()


# Process-wide shared store singleton. Used so low-level adapters (e.g. the
# compression telemetry sink) can reach the same SQLite-backed store the server
# uses, without importing server.py (which would be a circular import).
_default_store: "SkillStore | None" = None


def get_store() -> "SkillStore":
    """Return the process-wide :class:`SkillStore`, creating it on first call."""
    global _default_store
    if _default_store is None:
        _default_store = SkillStore()
    return _default_store
