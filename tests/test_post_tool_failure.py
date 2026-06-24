"""Tests for post_tool_observer.py PostToolUseFailure branch (Claude Code 2.1.119)
and _maybe_emit_skill_used (issue #94).
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

HOOKS_DIR = Path(__file__).resolve().parent.parent / "hooks"
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(HOOKS_DIR))
sys.path.insert(0, str(SRC_DIR))


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


# ===========================================================================
# _maybe_emit_skill_used — issue #94
# ===========================================================================


class TestMaybeEmitSkillUsed:
    """Tests for post_tool_observer._maybe_emit_skill_used."""

    def _make_store(self, tmp_path, name: str = "hook_test.db"):
        from skill_hub.store import SkillStore
        store = SkillStore(db_path=tmp_path / name)
        store.close = lambda: None  # keep open for assertions
        return store

    def test_skill_used_event_emitted_on_search_skills(self, tmp_path):
        """search_skills PostToolUse with LOADED skills emits skill.used events."""
        import post_tool_observer as pto
        import skill_hub.store as _s_mod

        store = self._make_store(tmp_path, "hook1.db")
        response_text = (
            "<!-- Skill Hub search: query='test' top_k=3 mode=vector -->\n"
            "<!-- LOADED (2): skill-alpha, skill-beta -->\n"
            "<!-- NOT LOADED (1): skill-gamma -->\n"
        )

        with patch.object(_s_mod, "SkillStore", return_value=store):
            pto._maybe_emit_skill_used(
                pto._SEARCH_SKILLS_TOOL, response_text, "sess-hook-1"
            )

        rows = store.get_events(session_id="sess-hook-1", kind="skill.used")
        assert len(rows) == 2
        skill_ids = {json.loads(r["payload"])["skill_id"] for r in rows}
        assert skill_ids == {"skill-alpha", "skill-beta"}

    def test_matched_true_when_prior_injection_exists(self, tmp_path):
        """injection_id is resolved when a skill_injections row exists."""
        import post_tool_observer as pto
        import skill_hub.store as _s_mod
        from skill_hub.store import SkillStore

        store = SkillStore(db_path=tmp_path / "match_test.db")
        # Pre-populate an injection row
        store.log_skill_injection("skill-match", query="q", session_id="sess-m")
        store.close = lambda: None  # keep open for assertions

        with patch.object(_s_mod, "SkillStore", return_value=store):
            pto._maybe_emit_skill_used(
                tool_name=pto._SEARCH_SKILLS_TOOL,
                tool_response=(
                    "<!-- LOADED (1): skill-match -->\n"
                    "<!-- NOT LOADED (0): none -->\n"
                ),
                session_id="sess-m",
            )

        rows = store.get_events(session_id="sess-m", kind="skill.used")
        assert len(rows) == 1
        payload = json.loads(rows[0]["payload"])
        assert payload["matched"] is True
        assert payload["injection_id"] is not None

    def test_no_event_for_non_search_skills_tool(self, tmp_path):
        """Non-search_skills tools must not emit skill.used events."""
        import post_tool_observer as pto
        import skill_hub.store as _s_mod

        store = self._make_store(tmp_path, "bash_test.db")

        with patch.object(_s_mod, "SkillStore", return_value=store):
            pto._maybe_emit_skill_used("Bash", "some output", "sess-bash")

        rows = store.get_events(session_id="sess-bash", kind="skill.used")
        assert rows == []

    def test_no_event_when_loaded_is_none(self, tmp_path):
        """Response with 'none' in LOADED list must produce no events."""
        import post_tool_observer as pto
        import skill_hub.store as _s_mod

        store = self._make_store(tmp_path, "none_test.db")
        response_text = "<!-- LOADED (0): none -->\n"

        with patch.object(_s_mod, "SkillStore", return_value=store):
            pto._maybe_emit_skill_used(
                pto._SEARCH_SKILLS_TOOL, response_text, "sess-none"
            )

        rows = store.get_events(session_id="sess-none", kind="skill.used")
        assert rows == []
