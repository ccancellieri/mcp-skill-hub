"""Tests for issue #127 — task lifecycle residual gaps.

Covers:
1. session_start_enforcer._ensure_open_tasks: title derivation from the linked
   memory file's frontmatter description (not the raw filename slug), skip
   when no description, dedupe against already-open task titles.
   Plus store.find_junk_memory_tasks / cleanup_junk_memory_tasks.
2. store.bind_task_to_session / save_session_context: the session_context
   .task_id backlink (additive migration + backfill + write-both-sides).
3. base_config.py / install.py: the PostToolUse hook must not carry the
   Bash(*) filter that starved the Claude /tasks projection of TodoWrite /
   TaskCreate / TaskUpdate / TaskComplete / TaskStop calls.

Constraints (hard)
------------------
- Every test uses a fresh tmp_path DB, NEVER the live DB_PATH.
- Any test touching skill_hub.config monkeypatches cfg.CONFIG_PATH.
- Never imports skill_hub.server.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
HOOKS = Path(__file__).resolve().parent.parent / "hooks"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HOOKS))


def test_server_not_imported(assert_server_not_imported):  # noqa: PT019
    """skill_hub.server must not be imported at collection time — it opens the live DB."""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path):
    from skill_hub.store import SkillStore
    s = SkillStore(db_path=tmp_path / "issue127_test.db")
    yield s
    s.close()


def _import_ensure_open_tasks():
    """Import session_start_enforcer._ensure_open_tasks, stubbing verdict_cache
    (it is imported at module scope by session_start_enforcer.py and pulls in
    machinery unrelated to this hook)."""
    import types
    if "verdict_cache" not in sys.modules:
        vc = types.ModuleType("verdict_cache")
        vc.load_config = lambda: {}  # type: ignore[attr-defined]
        vc.connect = lambda: None  # type: ignore[attr-defined]
        vc.task_tag = lambda: ""  # type: ignore[attr-defined]
        sys.modules["verdict_cache"] = vc
    import session_start_enforcer as sse
    return sse


# ---------------------------------------------------------------------------
# A) _ensure_open_tasks — title derivation + skip + dedupe (defect 1)
# ---------------------------------------------------------------------------

class TestEnsureOpenTasksTitleFix:
    def _setup(self, tmp_path, monkeypatch, store):
        monkeypatch.setenv("HOME", str(tmp_path))
        from skill_hub import config as _cfg
        monkeypatch.setattr(_cfg, "CONFIG_PATH", tmp_path / "config.json")

        sse = _import_ensure_open_tasks()

        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        memory_index = memory_dir / "MEMORY.md"

        store.close = lambda: None  # keep open for assertions
        import skill_hub.store as _s_mod
        monkeypatch.setattr(_s_mod, "SkillStore", lambda: store)
        monkeypatch.setattr(sse, "_find_memory_index", lambda: memory_index)

        return sse, memory_dir, memory_index

    def test_title_uses_frontmatter_description_not_slug(self, tmp_path, monkeypatch, store):
        """A memory file WITH frontmatter must yield a description-based title,
        never the filename slug (the pre-fix junk-title shape)."""
        sse, memory_dir, memory_index = self._setup(tmp_path, monkeypatch, store)

        (memory_dir / "project_search_harmonization_1777.md").write_text(
            "---\n"
            "name: project_search_harmonization_1777\n"
            "description: GeoID search-surface harmonization work\n"
            "---\n\nbody\n"
        )
        memory_index.write_text(
            "# Memory Index\n\n"
            "- [project_search_harmonization_1777.md]"
            "(project_search_harmonization_1777.md) — PARTIAL, sortby remains.\n"
        )

        msg = sse._ensure_open_tasks()
        assert "created 1 open task" in msg

        rows = store._conn.execute("SELECT title, tags FROM tasks").fetchall()
        assert len(rows) == 1
        assert rows[0]["title"] == "GeoID search-surface harmonization work"
        assert rows[0]["title"] != "project_search_harmonization_1777"

    def test_entry_without_description_is_skipped(self, tmp_path, monkeypatch, store):
        """A frontmatter-less (or description-less) memory file must be
        skipped entirely — never falls back to the raw filename slug."""
        sse, memory_dir, memory_index = self._setup(tmp_path, monkeypatch, store)

        (memory_dir / "project_no_frontmatter.md").write_text("just a body, no frontmatter\n")
        memory_index.write_text(
            "# Memory Index\n\n"
            "- [project_no_frontmatter.md](project_no_frontmatter.md) — PARTIAL work remains.\n"
        )

        msg = sse._ensure_open_tasks()
        assert msg == ""
        count = store._conn.execute("SELECT count(*) FROM tasks").fetchone()[0]
        assert count == 0

    def test_dedupes_against_existing_open_task_title(self, tmp_path, monkeypatch, store):
        """A description matching an already-open task's title must not
        spawn a duplicate task."""
        sse, memory_dir, memory_index = self._setup(tmp_path, monkeypatch, store)

        store.save_task(title="Shared harmonization effort", summary="", vector=[])

        (memory_dir / "project_dup.md").write_text(
            "---\ndescription: Shared harmonization effort\n---\n\nbody\n"
        )
        memory_index.write_text(
            "# Memory Index\n\n"
            "- [project_dup.md](project_dup.md) — PARTIAL, dup check.\n"
        )

        msg = sse._ensure_open_tasks()
        assert msg == ""
        count = store._conn.execute("SELECT count(*) FROM tasks").fetchone()[0]
        assert count == 1


# ---------------------------------------------------------------------------
# B) store.find_junk_memory_tasks / cleanup_junk_memory_tasks
# ---------------------------------------------------------------------------

class TestCleanupJunkMemoryTasks:
    def test_finds_only_slug_shaped_sessionless_memory_tasks(self, store):
        # Junk: slug title, src:memory tag, no session.
        junk_id = store.save_task(
            title="project_search_harmonization_1777", summary="", vector=[],
            tags="src:memory",
        )
        # Legit: human description title, src:memory tag, no session.
        store.save_task(
            title="GeoID search-surface harmonization work", summary="", vector=[],
            tags="src:memory",
        )
        # Slug-shaped but has a session — must NOT be flagged.
        store.save_task(
            title="project_should_survive_1", summary="", vector=[],
            tags="src:memory", session_id="sess-1",
        )
        # Slug-shaped but not src:memory — must NOT be flagged.
        store.save_task(
            title="not_memory_tagged_task", summary="", vector=[],
            tags="src:claude-task",
        )

        junk = store.find_junk_memory_tasks()
        assert [int(r["id"]) for r in junk] == [junk_id]

    def test_dry_run_does_not_delete(self, store):
        tid = store.save_task(
            title="project_dry_run_junk", summary="", vector=[], tags="src:memory",
        )
        report = store.cleanup_junk_memory_tasks(dry_run=True)
        assert report["dry_run"] is True
        assert report["count"] == 1
        row = store._conn.execute("SELECT id FROM tasks WHERE id=?", (tid,)).fetchone()
        assert row is not None

    def test_apply_deletes_and_is_idempotent(self, store):
        tid = store.save_task(
            title="project_apply_junk", summary="", vector=[], tags="src:memory",
        )
        report = store.cleanup_junk_memory_tasks(dry_run=False)
        assert report["count"] == 1
        row = store._conn.execute("SELECT id FROM tasks WHERE id=?", (tid,)).fetchone()
        assert row is None

        # Second run: no-op.
        report2 = store.cleanup_junk_memory_tasks(dry_run=False)
        assert report2["count"] == 0
        assert report2["removed"] == []


# ---------------------------------------------------------------------------
# C) session<->task backlink (defect 2)
# ---------------------------------------------------------------------------

class TestSessionTaskBacklink:
    def test_task_id_column_exists_on_session_context(self, store):
        cols = {row[1] for row in store._conn.execute("PRAGMA table_info(session_context)")}
        assert "task_id" in cols

    def test_bind_task_to_session_writes_both_sides(self, store):
        tid = store.save_task(title="Bindable work", summary="", vector=[])
        store.save_session_context(
            session_id="sess-bind", loaded_skills=[], context_summary="",
            message_count=0,
        )
        store.bind_task_to_session(tid, "sess-bind")

        task_row = store._conn.execute(
            "SELECT session_id FROM tasks WHERE id=?", (tid,)
        ).fetchone()
        assert task_row["session_id"] == "sess-bind"

        ctx = store.get_session_context("sess-bind")
        assert ctx["task_id"] == tid

    def test_save_session_context_self_heals_task_id_on_first_insert(self, store):
        """A task created for a session BEFORE session_context exists must
        still surface via task_id once the context row is inserted (the
        original ordering bug: session_context is often INSERTed for the
        first time only after bind_task_to_session already ran)."""
        tid = store.save_task(
            title="Pre-context work", summary="", vector=[], session_id="sess-late",
        )
        # session_context row does not exist yet — first insert now.
        store.save_session_context(
            session_id="sess-late", loaded_skills=[], context_summary="",
            message_count=1,
        )
        ctx = store.get_session_context("sess-late")
        assert ctx["task_id"] == tid

    def test_save_session_context_does_not_clobber_task_id_on_update(self, store):
        tid = store.save_task(title="Sticky task", summary="", vector=[])
        store.save_session_context(
            session_id="sess-sticky", loaded_skills=[], context_summary="",
            message_count=0,
        )
        store.bind_task_to_session(tid, "sess-sticky")

        # A later context update (e.g. a new message) must not wipe task_id.
        store.save_session_context(
            session_id="sess-sticky", loaded_skills=["a"], context_summary="updated",
            message_count=2,
        )
        ctx = store.get_session_context("sess-sticky")
        assert ctx["task_id"] == tid

    def test_migration_backfills_existing_session_task_matches(self, tmp_path):
        """A pre-#127 DB (session_context predates task_id) must have its
        existing tasks.session_id matches backfilled on the next open."""
        import sqlite3
        db_path = tmp_path / "legacy.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL, summary TEXT NOT NULL, context TEXT,
                status TEXT NOT NULL DEFAULT 'open', tags TEXT, compact TEXT,
                vector TEXT, session_id TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')), closed_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE session_context (
                session_id TEXT PRIMARY KEY,
                loaded_skills TEXT NOT NULL DEFAULT '[]',
                context_summary TEXT NOT NULL DEFAULT '',
                message_count INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute(
            "INSERT INTO tasks (title, summary, session_id, status) "
            "VALUES ('Legacy work', '', 'sess-legacy', 'open')"
        )
        conn.execute(
            "INSERT INTO session_context (session_id) VALUES ('sess-legacy')"
        )
        conn.commit()
        conn.close()

        from skill_hub.store import SkillStore
        s = SkillStore(db_path=db_path)
        try:
            ctx = s.get_session_context("sess-legacy")
            tid = s._conn.execute(
                "SELECT id FROM tasks WHERE session_id='sess-legacy'"
            ).fetchone()["id"]
            assert ctx["task_id"] == tid
        finally:
            s.close()


# ---------------------------------------------------------------------------
# D) PostToolUse hook registration must not filter out task-tool calls (defect 3)
# ---------------------------------------------------------------------------

class TestPostToolUseHookNotBashFiltered:
    def test_base_config_post_tool_use_has_no_if_filter(self):
        from skill_hub import base_config as bc
        hooks = bc.base_hooks()
        ptu = hooks["PostToolUse"][0]["hooks"][0]
        assert "post-tool-observer.sh" in ptu["command"]
        assert "if" not in ptu, (
            "PostToolUse must fire on every tool call — TodoWrite/TaskCreate/"
            "TaskUpdate/TaskComplete/TaskStop calls need to reach the observer "
            "for the /tasks projection to ingest anything (#127)"
        )

    def test_base_config_post_tool_use_failure_keeps_bash_filter(self):
        """PostToolUseFailure only feeds the Bash verdict cache — Bash(*) stays."""
        from skill_hub import base_config as bc
        hooks = bc.base_hooks()
        ptuf = hooks["PostToolUseFailure"][0]["hooks"][0]
        assert ptuf.get("if") == "Bash(*)"

    def test_install_py_post_tool_use_has_no_if_filter(self, monkeypatch, tmp_path):
        import importlib.util
        root = Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location("install_under_test_127", root / "install.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["install_under_test_127"] = mod
        spec.loader.exec_module(mod)
        settings_path = tmp_path / "settings.json"
        monkeypatch.setattr(mod, "SETTINGS", settings_path)
        mod.step_install_hooks(1, 1)

        import json
        data = json.loads(settings_path.read_text())
        ptu_hooks = [h for entry in data["hooks"]["PostToolUse"] for h in entry["hooks"]]
        observer = [h for h in ptu_hooks if "post-tool-observer.sh" in h["command"]]
        assert observer, "post-tool-observer.sh missing from PostToolUse"
        assert "if" not in observer[0]

        ptuf_hooks = [h for entry in data["hooks"]["PostToolUseFailure"] for h in entry["hooks"]]
        assert ptuf_hooks[0].get("if") == "Bash(*)"


# ---------------------------------------------------------------------------
# E) Ingest end-to-end sanity: once the hook fires, TodoWrite reaches project_claude_task
# ---------------------------------------------------------------------------

class TestClaudeTaskIngestFiresForTodoWrite:
    def test_maybe_observe_claude_task_creates_row_for_todowrite(self, tmp_path):
        """Confirms the ingest CODE path itself is sound — the only thing
        blocking it in production was the PostToolUse Bash(*) filter (see
        TestPostToolUseHookNotBashFiltered above)."""
        import post_tool_observer as pto
        import skill_hub.store as _s_mod
        from skill_hub.store import SkillStore

        store = SkillStore(db_path=tmp_path / "ingest_test.db")
        store.close = lambda: None  # keep open for assertions

        with patch.object(_s_mod, "SkillStore", return_value=store):
            pto._maybe_observe_claude_task({
                "session_id": "sess-todo",
                "cwd": str(tmp_path),
                "tool_name": "TodoWrite",
                "tool_input": {
                    "todos": [
                        {"content": "Write the fix", "status": "in_progress"},
                    ]
                },
                "tool_response": {},
            })

        rows = store._conn.execute(
            "SELECT title, claude_task_key FROM tasks WHERE tags LIKE '%src:claude-task%'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["title"] == "Write the fix"
        assert rows[0]["claude_task_key"] is not None
