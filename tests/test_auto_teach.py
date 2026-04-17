"""Tests for Phase G.2 continuous teaching features.

Covers:
- _maybe_auto_teach_from_feedback (post_tool_observer)
- _maybe_teach_from_message (session_start_enforcer)
- touch_task_activity / get_task_activity_state / list_tasks_with_activity (store)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

SRC = Path(__file__).resolve().parent.parent / "src"
HOOKS = Path(__file__).resolve().parent.parent / "hooks"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HOOKS))

import pytest


# ---------------------------------------------------------------------------
# Store fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path, monkeypatch):
    from skill_hub.store import SkillStore
    monkeypatch.setenv("HOME", str(tmp_path))
    s = SkillStore(db_path=tmp_path / "skill_hub.db")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Store — last_activity_at / heartbeat tests
# ---------------------------------------------------------------------------

def test_last_activity_at_column_exists(store):
    """last_activity_at column must be present after migration."""
    cols = {row[1] for row in store._conn.execute("PRAGMA table_info(tasks)")}
    assert "last_activity_at" in cols


def test_touch_task_activity_updates_timestamp(store):
    """touch_task_activity sets last_activity_at to a non-null value."""
    task_id = store.save_task(
        title="heartbeat test",
        summary="check heartbeat",
        vector=[],
        session_id="sess-hb",
    )
    # Before touch: should be None
    row = store._conn.execute(
        "SELECT last_activity_at FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    assert row["last_activity_at"] is None

    store.touch_task_activity(task_id)

    row = store._conn.execute(
        "SELECT last_activity_at FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    assert row["last_activity_at"] is not None


def test_get_task_activity_state_open_no_heartbeat(store):
    """An open task with no heartbeat returns 'open'."""
    task_id = store.save_task(
        title="no heartbeat",
        summary="nothing",
        vector=[],
        session_id="sess-x",
    )
    state = store.get_task_activity_state(task_id)
    assert state == "open"


def test_get_task_activity_state_active_after_touch(store):
    """A task touched just now returns 'active'."""
    task_id = store.save_task(
        title="active task",
        summary="just started",
        vector=[],
        session_id="sess-a",
    )
    store.touch_task_activity(task_id)
    state = store.get_task_activity_state(task_id)
    assert state == "active"


def test_get_task_activity_state_closed(store):
    """A closed task always returns 'closed'."""
    task_id = store.save_task(
        title="closed task",
        summary="done",
        vector=[],
        session_id="sess-c",
    )
    store.close_task(task_id, compact="finished", compact_vector=None)
    state = store.get_task_activity_state(task_id)
    assert state == "closed"


def test_get_task_activity_state_unknown(store):
    """Non-existent task id returns 'unknown'."""
    state = store.get_task_activity_state(99999)
    assert state == "unknown"


def test_get_open_task_id_for_session(store):
    """get_open_task_id_for_session returns int id for an open task."""
    task_id = store.save_task(
        title="session task",
        summary="test",
        vector=[],
        session_id="sess-q",
    )
    result = store.get_open_task_id_for_session("sess-q")
    assert result == task_id
    assert isinstance(result, int)


def test_get_open_task_id_for_session_none_when_empty(store):
    """Returns None when no open task for session."""
    result = store.get_open_task_id_for_session("nonexistent-session")
    assert result is None


def test_list_tasks_with_activity_includes_state(store):
    """list_tasks_with_activity includes activity_state for each task."""
    store.save_task(title="task A", summary="a", vector=[], session_id="s1")
    rows = store.list_tasks_with_activity()
    assert len(rows) >= 1
    for r in rows:
        assert "activity_state" in r
        assert r["activity_state"] in ("active", "idle", "open", "closed")


def test_list_tasks_with_activity_filters_by_status(store):
    """list_tasks_with_activity respects the status filter."""
    t1 = store.save_task(title="open task", summary="open", vector=[], session_id="s2")
    t2 = store.save_task(title="closed task", summary="closed", vector=[], session_id="s3")
    store.close_task(t2, compact="done", compact_vector=None)

    open_rows = store.list_tasks_with_activity(status="open")
    closed_rows = store.list_tasks_with_activity(status="closed")

    open_ids = [r["id"] for r in open_rows]
    closed_ids = [r["id"] for r in closed_rows]
    assert t1 in open_ids
    assert t2 not in open_ids
    assert t2 in closed_ids


# ---------------------------------------------------------------------------
# post_tool_observer — _maybe_auto_teach_from_feedback
# ---------------------------------------------------------------------------

def _import_auto_teach():
    """Import _maybe_auto_teach_from_feedback from post_tool_observer."""
    # Insert hooks dir and a stub verdict_cache so we can import the hook
    import importlib
    import types
    if "verdict_cache" not in sys.modules:
        vc = types.ModuleType("verdict_cache")
        vc.load_config = lambda: {}  # type: ignore[attr-defined]
        vc.connect = lambda: None  # type: ignore[attr-defined]
        vc.put = lambda *a, **kw: None  # type: ignore[attr-defined]
        vc.task_tag = lambda: ""  # type: ignore[attr-defined]
        sys.modules["verdict_cache"] = vc
    import post_tool_observer
    return post_tool_observer._maybe_auto_teach_from_feedback


def test_auto_teach_skips_non_feedback_file():
    """_maybe_auto_teach_from_feedback does nothing for non-feedback filenames."""
    fn = _import_auto_teach()
    # Should not raise and should not call store
    fn("Write", {"file_path": "/some/path/regular_notes.md"})


def test_auto_teach_skips_non_write_tool():
    """_maybe_auto_teach_from_feedback skips non-Write/Edit tools."""
    fn = _import_auto_teach()
    fn("Bash", {"command": "echo feedback_test.md"})


def test_auto_teach_skips_when_disabled(tmp_path, monkeypatch):
    """_maybe_auto_teach_from_feedback does nothing when continuous_teaching_enabled=False."""
    fn = _import_auto_teach()
    # continuous_teaching_enabled defaults to False — should skip even with valid file
    feedback_file = tmp_path / "feedback_test_rule.md"
    feedback_file.write_text("Always run tests before committing.\n\n**Why:** CI depends on it.")
    fn("Write", {"file_path": str(feedback_file)})  # should not raise


def test_auto_teach_extracts_rule(tmp_path, monkeypatch):
    """_maybe_auto_teach_from_feedback stores a teaching when enabled."""
    # Enable continuous teaching
    monkeypatch.setenv("HOME", str(tmp_path))
    from skill_hub import config as _cfg
    _cfg._DEFAULTS["continuous_teaching_enabled"] = True

    from skill_hub.store import SkillStore
    store = SkillStore(db_path=tmp_path / "skill_hub.db")

    # Wrap store so close() is a no-op — the hook calls store.close() internally
    # but we need to keep the connection open for assertions.
    _real_close = store.close
    store.close = lambda: None  # type: ignore[method-assign]

    feedback_file = tmp_path / "feedback_auto_teach.md"
    feedback_file.write_text(
        "Always run tests before committing.\n\n**Why:** CI depends on it.\n"
    )

    # Patch SkillStore so _maybe_auto_teach_from_feedback uses our tmp store
    import post_tool_observer as pto
    try:
        import skill_hub.store as _sh_store

        def _fake_store(*args, **kwargs):
            return store

        with patch.object(_sh_store, "SkillStore", side_effect=_fake_store):
            pto._maybe_auto_teach_from_feedback("Write", {"file_path": str(feedback_file)})

        rows = store._conn.execute("SELECT rule FROM teachings").fetchall()
        rules = [r[0] for r in rows]
        assert any("Always run tests" in r for r in rules), f"No matching rule found in {rules}"
    finally:
        _cfg._DEFAULTS["continuous_teaching_enabled"] = False
        _real_close()


# ---------------------------------------------------------------------------
# session_start_enforcer — _maybe_teach_from_message
# ---------------------------------------------------------------------------

def _import_teach_from_message():
    """Import _maybe_teach_from_message from session_start_enforcer."""
    import types
    if "verdict_cache" not in sys.modules:
        vc = types.ModuleType("verdict_cache")
        vc.load_config = lambda: {}  # type: ignore[attr-defined]
        vc.connect = lambda: None  # type: ignore[attr-defined]
        vc.task_tag = lambda: ""  # type: ignore[attr-defined]
        sys.modules["verdict_cache"] = vc
    import session_start_enforcer
    return session_start_enforcer._maybe_teach_from_message


def test_teach_from_message_empty_returns_empty():
    """_maybe_teach_from_message returns '' for empty message."""
    fn = _import_teach_from_message()
    result = fn("", "sess-1")
    assert result == ""


def test_teach_from_message_non_teach_returns_empty():
    """_maybe_teach_from_message returns '' for non-teach messages."""
    fn = _import_teach_from_message()
    result = fn("Can you help me fix this bug?", "sess-2")
    assert result == ""


def test_teach_from_message_disabled_by_config():
    """_maybe_teach_from_message returns '' when continuous_teaching_enabled=False."""
    fn = _import_teach_from_message()
    # Default is False — should return empty even for matching patterns
    result = fn("remember: never commit secrets to repos", "sess-3")
    assert result == ""


def test_teach_from_message_remember_pattern(tmp_path, monkeypatch):
    """_maybe_teach_from_message stores teaching when enabled and pattern matches."""
    monkeypatch.setenv("HOME", str(tmp_path))
    from skill_hub import config as _cfg
    _cfg._DEFAULTS["continuous_teaching_enabled"] = True

    from skill_hub.store import SkillStore
    store = SkillStore(db_path=tmp_path / "skill_hub.db")

    # Wrap close() to be a no-op so we can inspect teachings after the hook runs
    _real_close = store.close
    store.close = lambda: None  # type: ignore[method-assign]

    import session_start_enforcer as sse
    import skill_hub.store as _sh_store

    def _fake_store(*args, **kwargs):
        return store

    try:
        with patch.object(_sh_store, "SkillStore", side_effect=_fake_store):
            result = sse._maybe_teach_from_message(
                "remember: never commit API keys to public repos", "sess-teach"
            )

        assert result.startswith('AUTO-TAUGHT:')
        rows = store._conn.execute("SELECT rule FROM teachings").fetchall()
        rules = [r[0] for r in rows]
        assert any("API keys" in r or "never commit" in r.lower() for r in rules), \
            f"No matching rule found in {rules}"
    finally:
        _cfg._DEFAULTS["continuous_teaching_enabled"] = False
        _real_close()
