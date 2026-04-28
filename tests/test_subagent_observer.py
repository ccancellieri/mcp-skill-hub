"""Tests for hooks/subagent_observer.py + the session_log schema migration."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

HOOKS_DIR = Path(__file__).resolve().parent.parent / "hooks"
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(HOOKS_DIR))
sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    from skill_hub.store import SkillStore
    db_path = tmp_path / "skill_hub.db"
    monkeypatch.setattr("skill_hub.store.DB_PATH", db_path)
    store = SkillStore(db_path=db_path)
    yield store
    store.close()


def test_session_log_has_agent_columns(isolated_store):
    """The migration must add agent_id/agent_type/event/transcript_path."""
    cols = {row[1] for row in isolated_store._conn.execute(
        "PRAGMA table_info(session_log)")}
    for c in ("agent_id", "agent_type", "event", "transcript_path"):
        assert c in cols, f"missing column {c} in session_log: {cols}"


def test_log_session_subagent_persists_event(isolated_store):
    isolated_store.log_session_subagent(
        session_id="sess-abc",
        agent_id="agent-xyz",
        agent_type="Explore",
        event="SubagentStart",
        transcript_path="/tmp/xyz.jsonl",
    )
    row = isolated_store._conn.execute(
        "SELECT session_id, agent_id, agent_type, event, transcript_path, tool_used "
        "FROM session_log WHERE agent_id = ?", ("agent-xyz",),
    ).fetchone()
    assert row is not None
    assert row["session_id"] == "sess-abc"
    assert row["agent_type"] == "Explore"
    assert row["event"] == "SubagentStart"
    assert row["transcript_path"] == "/tmp/xyz.jsonl"
    # tool_used column must be filled to keep existing aggregations happy.
    assert row["tool_used"] == "subagent"


def test_log_session_tool_still_works(isolated_store):
    """Schema migration must not break the existing log_session_tool path."""
    isolated_store.log_session_tool(
        session_id="sess-abc",
        query="hello",
        query_vector=None,
        tool_used="search_skills",
        plugin_id="skill-hub",
    )
    row = isolated_store._conn.execute(
        "SELECT tool_used, plugin_id, agent_id "
        "FROM session_log WHERE session_id = ?", ("sess-abc",),
    ).fetchone()
    assert row["tool_used"] == "search_skills"
    assert row["plugin_id"] == "skill-hub"
    assert row["agent_id"] is None


def test_observer_logs_subagent_start(isolated_store, monkeypatch):
    """End-to-end: hook stdin → log_session_subagent row in session_log."""
    # Patch SkillStore in skill_hub.store to return our isolated instance.
    import skill_hub.store as _ss
    monkeypatch.setattr(_ss, "SkillStore", lambda *a, **kw: isolated_store)
    # Prevent the observer from calling .close() and breaking the fixture.
    monkeypatch.setattr(isolated_store, "close", lambda: None)

    import subagent_observer
    real_stdin = sys.stdin
    sys.stdin = io.StringIO(json.dumps({
        "hook_event_name": "SubagentStart",
        "session_id": "s-1",
        "agent_id": "a-1",
        "agent_type": "Plan",
    }))
    try:
        rc = subagent_observer.main()
    finally:
        sys.stdin = real_stdin

    assert rc == 0
    row = isolated_store._conn.execute(
        "SELECT agent_type, event FROM session_log WHERE agent_id = ?",
        ("a-1",),
    ).fetchone()
    assert row is not None
    assert row["agent_type"] == "Plan"
    assert row["event"] == "SubagentStart"


def test_observer_skips_when_missing_ids(isolated_store, monkeypatch):
    import skill_hub.store as _ss
    monkeypatch.setattr(_ss, "SkillStore", lambda *a, **kw: isolated_store)

    import subagent_observer
    real_stdin = sys.stdin
    sys.stdin = io.StringIO(json.dumps({
        "hook_event_name": "SubagentStart",
        # missing session_id and agent_id
    }))
    try:
        rc = subagent_observer.main()
    finally:
        sys.stdin = real_stdin

    assert rc == 0
    rows = isolated_store._conn.execute(
        "SELECT * FROM session_log").fetchall()
    assert rows == []
