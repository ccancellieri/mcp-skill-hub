"""Tests for post_tool_observer.py PostToolUseFailure branch (Claude Code 2.1.119)."""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import pytest

HOOKS_DIR = Path(__file__).resolve().parent.parent / "hooks"
sys.path.insert(0, str(HOOKS_DIR))


@pytest.fixture
def isolated_verdict_db(tmp_path, monkeypatch):
    """Point verdict_cache at a temp DB so tests don't touch the real cache."""
    import verdict_cache
    monkeypatch.setattr(verdict_cache, "DB_PATH", tmp_path / "verdicts.db")
    monkeypatch.setattr(verdict_cache, "CONFIG_PATH",
                        tmp_path / "config.json")
    # Pre-init the schema in the temp DB.
    conn = verdict_cache.connect()
    conn.close()
    yield verdict_cache


def _run_with_input(payload: dict) -> int:
    """Run post_tool_observer.main() with payload as stdin."""
    import post_tool_observer
    real_stdin = sys.stdin
    sys.stdin = io.StringIO(json.dumps(payload))
    try:
        return post_tool_observer.main()
    finally:
        sys.stdin = real_stdin


def test_failure_event_records_failed_verdict(isolated_verdict_db):
    """A PostToolUseFailure for a Bash tool stores `decision=failed`."""
    rc = _run_with_input({
        "session_id": "s-1",
        "hook_event_name": "PostToolUseFailure",
        "tool_name": "Bash",
        "tool_input": {"command": "false"},
        "duration_ms": 12,
    })
    assert rc == 0

    conn = isolated_verdict_db.connect()
    rows = conn.execute(
        "SELECT decision, source FROM command_verdicts").fetchall()
    assert len(rows) == 1
    assert rows[0]["decision"] == "failed"
    assert rows[0]["source"] == "tool_failure"


def test_failure_event_skips_non_bash_tools(isolated_verdict_db):
    """Non-Bash tool failures are ignored (the `if` filter skips us anyway,
    but the script is defensive)."""
    rc = _run_with_input({
        "session_id": "s-1",
        "hook_event_name": "PostToolUseFailure",
        "tool_name": "Edit",
        "tool_input": {"file_path": "/tmp/foo"},
    })
    assert rc == 0

    conn = isolated_verdict_db.connect()
    rows = conn.execute("SELECT * FROM command_verdicts").fetchall()
    assert rows == []


def test_user_approved_survives_a_failure(isolated_verdict_db):
    """A previously user-approved command should NOT be downgraded to failed
    after one transient failure (priority semantics)."""
    vc = isolated_verdict_db
    conn = vc.connect()
    vc.put(conn, "Bash", "git status", "allow", "user_approved", 1.0)

    rc = _run_with_input({
        "session_id": "s-1",
        "hook_event_name": "PostToolUseFailure",
        "tool_name": "Bash",
        "tool_input": {"command": "git status"},
    })
    assert rc == 0

    conn = vc.connect()
    rows = conn.execute(
        "SELECT decision, source FROM command_verdicts").fetchall()
    # Still user_approved; the failure was rejected by priority guard.
    assert len(rows) == 1
    assert rows[0]["decision"] == "allow"
    assert rows[0]["source"] == "user_approved"


def test_failure_overrides_llm_verdict(isolated_verdict_db):
    """An LLM-classified `allow` should be overwritten by a real-world failure."""
    vc = isolated_verdict_db
    conn = vc.connect()
    vc.put(conn, "Bash", "rm /tmp/maybe-missing", "allow", "llm", 0.8)

    rc = _run_with_input({
        "session_id": "s-1",
        "hook_event_name": "PostToolUseFailure",
        "tool_name": "Bash",
        "tool_input": {"command": "rm /tmp/maybe-missing"},
    })
    assert rc == 0

    conn = vc.connect()
    rows = conn.execute(
        "SELECT decision, source FROM command_verdicts").fetchall()
    assert len(rows) == 1
    assert rows[0]["decision"] == "failed"
    assert rows[0]["source"] == "tool_failure"


def test_post_tool_use_success_still_records_allow(isolated_verdict_db):
    """Make sure the PostToolUse (success) branch still works after the refactor."""
    rc = _run_with_input({
        "session_id": "s-1",
        "hook_event_name": "PostToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": "echo hi"},
        "tool_response": {"stdout": "hi"},
    })
    assert rc == 0

    conn = isolated_verdict_db.connect()
    rows = conn.execute(
        "SELECT decision, source FROM command_verdicts").fetchall()
    assert len(rows) == 1
    assert rows[0]["decision"] == "allow"
    assert rows[0]["source"] == "user_approved"
