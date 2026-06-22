"""Tests for issue #38 — Claude Code task ingestion into skill-hub.

Constraints (hard)
------------------
- Uses only skill_hub.store and skill_hub.claude_tasks — never skill_hub.server.
- Every test uses a fresh tmp_path DB, NEVER DB_PATH.
- No embedding model loaded (vector=[]).
- Do NOT import skill_hub.server anywhere in this file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Safety guard
# ---------------------------------------------------------------------------

def test_server_not_imported(assert_server_not_imported):  # noqa: PT019
    """skill_hub.server must not be imported at collection time — it opens the live DB."""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path):
    from skill_hub.store import SkillStore
    return SkillStore(db_path=tmp_path / "test_claude_ingest.db")


# ---------------------------------------------------------------------------
# A) parse_claude_tasks
# ---------------------------------------------------------------------------

class TestParseClaudeTasks:
    def test_todo_write_two_todos(self):
        from skill_hub.claude_tasks import parse_claude_tasks

        tool_input = {
            "todos": [
                {"content": "Write unit tests", "status": "in_progress"},
                {"content": "Deploy to staging", "status": "completed"},
            ]
        }
        result = parse_claude_tasks("TodoWrite", tool_input)

        assert len(result) == 2
        assert result[0]["title"] == "Write unit tests"
        assert result[0]["status"] == "in_progress"
        assert result[0]["claude_id"] is None
        assert result[1]["title"] == "Deploy to staging"
        assert result[1]["status"] == "completed"

    def test_todo_write_uses_description_fallback(self):
        from skill_hub.claude_tasks import parse_claude_tasks

        tool_input = {
            "todos": [
                {"description": "Fallback description", "status": "pending"},
            ]
        }
        result = parse_claude_tasks("TodoWrite", tool_input)
        assert len(result) == 1
        assert result[0]["title"] == "Fallback description"
        assert result[0]["status"] == "pending"

    def test_task_create_with_description(self):
        from skill_hub.claude_tasks import parse_claude_tasks

        tool_input = {"description": "Implement OAuth flow", "task_id": "task-abc"}
        result = parse_claude_tasks("TaskCreate", tool_input)

        assert len(result) == 1
        assert result[0]["title"] == "Implement OAuth flow"
        assert result[0]["status"] == "open"
        assert result[0]["claude_id"] == "task-abc"

    def test_task_create_id_from_response(self):
        from skill_hub.claude_tasks import parse_claude_tasks

        tool_input = {"description": "Some task"}
        tool_response = {"task_id": "resp-123"}
        result = parse_claude_tasks("TaskCreate", tool_input, tool_response)

        assert result[0]["claude_id"] == "resp-123"

    def test_task_complete_sets_completed_status(self):
        from skill_hub.claude_tasks import parse_claude_tasks

        tool_input = {"task_id": "task-xyz", "description": "Done task"}
        result = parse_claude_tasks("TaskComplete", tool_input)

        assert len(result) == 1
        assert result[0]["status"] == "completed"
        assert result[0]["claude_id"] == "task-xyz"

    def test_task_stop_sets_stopped_status(self):
        from skill_hub.claude_tasks import parse_claude_tasks

        tool_input = {"task_id": "task-stop"}
        result = parse_claude_tasks("TaskStop", tool_input)

        assert len(result) == 1
        assert result[0]["status"] == "stopped"
        assert result[0]["claude_id"] == "task-stop"

    def test_unknown_tool_returns_empty(self):
        from skill_hub.claude_tasks import parse_claude_tasks

        result = parse_claude_tasks("Bash", {"command": "ls"})
        assert result == []

    def test_unknown_tool_never_raises(self):
        from skill_hub.claude_tasks import parse_claude_tasks

        # Malformed input — must not raise.
        result = parse_claude_tasks("TodoWrite", {"todos": "not-a-list"})
        assert isinstance(result, list)

    def test_todo_write_empty_todos(self):
        from skill_hub.claude_tasks import parse_claude_tasks

        result = parse_claude_tasks("TodoWrite", {"todos": []})
        assert result == []

    def test_task_update_uses_custom_status(self):
        from skill_hub.claude_tasks import parse_claude_tasks

        tool_input = {"description": "Update me", "status": "in_progress", "task_id": "u1"}
        result = parse_claude_tasks("TaskUpdate", tool_input)
        assert result[0]["status"] == "in_progress"


# ---------------------------------------------------------------------------
# B) stable_key determinism
# ---------------------------------------------------------------------------

class TestStableKey:
    def test_same_identity_cwd_gives_same_key(self):
        from skill_hub.claude_tasks import stable_key

        k1 = stable_key("do the thing", cwd="/proj", branch="main")
        k2 = stable_key("do the thing", cwd="/proj", branch="main")
        assert k1 == k2

    def test_different_identity_gives_different_key(self):
        from skill_hub.claude_tasks import stable_key

        k1 = stable_key("task alpha", cwd="/proj", branch="main")
        k2 = stable_key("task beta", cwd="/proj", branch="main")
        assert k1 != k2

    def test_claude_id_gives_cid_form(self):
        from skill_hub.claude_tasks import stable_key

        k = stable_key("anything", cwd="/x", branch="main", claude_id="abc-123")
        assert k == "cid:abc-123"

    def test_no_claude_id_gives_txt_prefix(self):
        from skill_hub.claude_tasks import stable_key

        k = stable_key("my task", cwd="/home/user", branch="feat/x")
        assert k.startswith("txt:")
        assert len(k) == len("txt:") + 16

    def test_case_insensitive_normalization(self):
        from skill_hub.claude_tasks import stable_key

        k1 = stable_key("My Task", cwd="/proj", branch="main")
        k2 = stable_key("my task", cwd="/proj", branch="main")
        assert k1 == k2


# ---------------------------------------------------------------------------
# C) project_claude_task on tmp DB
# ---------------------------------------------------------------------------

class TestProjectClaudeTask:
    def test_create_new_task(self, store):
        result = store.project_claude_task(
            key="txt:abc123def456ab78",
            title="Implement feature X",
            status="open",
            session_id="sess-1",
        )
        assert result["action"] == "created"
        tid = result["task_id"]
        assert tid is not None

        # Verify the task exists in DB.
        row = store._conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        assert row is not None
        assert row["claude_task_key"] == "txt:abc123def456ab78"
        assert row["title"] == "Implement feature X"

    def test_re_project_same_key_is_updated_not_duplicated(self, store):
        """Projecting the same key twice (open) → update, still 1 task row."""
        key = "txt:aaaa1111bbbb2222"
        store.project_claude_task(
            key=key, title="Initial title", status="open", session_id="sess-1"
        )
        result = store.project_claude_task(
            key=key, title="Updated title", status="open", session_id="sess-1"
        )
        assert result["action"] == "updated"

        # Only one row in DB with this key.
        count = store._conn.execute(
            "SELECT count(*) FROM tasks WHERE claude_task_key = ?", (key,)
        ).fetchone()[0]
        assert count == 1

        # Title was updated.
        row = store._conn.execute(
            "SELECT title FROM tasks WHERE claude_task_key = ?", (key,)
        ).fetchone()
        assert row["title"] == "Updated title"

    def test_project_completion_closes_open_task(self, store):
        key = "cid:task-complete-test"
        # Create the task.
        store.project_claude_task(
            key=key, title="Task to complete", status="open", claude_id="task-complete-test"
        )
        # Complete it.
        result = store.project_claude_task(
            key=key, title="Task to complete", status="completed", claude_id="task-complete-test"
        )
        assert result["action"] == "closed"

        # Verify it is closed in DB.
        row = store._conn.execute(
            "SELECT status FROM tasks WHERE claude_task_key = ?", (key,)
        ).fetchone()
        assert row["status"] == "closed"

    def test_completion_on_missing_key_is_noop(self, store):
        result = store.project_claude_task(
            key="txt:nonexistent000000",
            title="Ghost task",
            status="completed",
        )
        assert result["action"] == "noop"

    def test_different_keys_create_distinct_tasks(self, store):
        store.project_claude_task(key="txt:key1111aaaa0000", title="Task A", status="open")
        store.project_claude_task(key="txt:key2222bbbb0000", title="Task B", status="open")

        count = store._conn.execute("SELECT count(*) FROM tasks").fetchone()[0]
        assert count == 2

    def test_claude_task_key_column_exists(self, store):
        """Schema migration: tasks table must have claude_task_key and claude_task_id."""
        cols = {r[1] for r in store._conn.execute("PRAGMA table_info(tasks)")}
        assert "claude_task_key" in cols
        assert "claude_task_id" in cols

    def test_unique_index_prevents_duplicate_keys(self, store):
        """Inserting two rows with the same claude_task_key must fail."""
        import sqlite3

        key = "txt:unique_test_key0"
        store.project_claude_task(key=key, title="First", status="open")
        with pytest.raises(sqlite3.IntegrityError):
            store._conn.execute(
                "INSERT INTO tasks (title, summary, vector, claude_task_key) "
                "VALUES (?, ?, ?, ?)",
                ("Second", "", "[]", key),
            )

    def test_cancelled_status_closes_task(self, store):
        key = "cid:cancel-me"
        store.project_claude_task(key=key, title="Cancel test", status="open", claude_id="cancel-me")
        result = store.project_claude_task(key=key, title="Cancel test", status="cancelled", claude_id="cancel-me")
        assert result["action"] == "closed"

    def test_stopped_status_closes_task(self, store):
        key = "cid:stop-me"
        store.project_claude_task(key=key, title="Stop test", status="open", claude_id="stop-me")
        result = store.project_claude_task(key=key, title="Stop test", status="stopped", claude_id="stop-me")
        assert result["action"] == "closed"
