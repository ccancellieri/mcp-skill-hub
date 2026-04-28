"""Tests for the new PreCompact / PostCompact / SessionEnd CLI verbs."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    from skill_hub.store import SkillStore
    db_path = tmp_path / "skill_hub.db"
    monkeypatch.setattr("skill_hub.store.DB_PATH", db_path)
    store = SkillStore(db_path=db_path)

    # The test environment has no embedder configured (no Ollama/Voyage/ST).
    # Replace upsert_vector with a thin INSERT into the vectors table so we
    # can assert on the CLI verb's logic without depending on embeddings.
    def _stub_upsert(namespace, doc_id, text, metadata=None, model=None,
                     level=None, source=None, project=None, tags=None):
        store._conn.execute(
            "INSERT OR REPLACE INTO vectors "
            "(namespace, doc_id, model, vector, norm, metadata, level, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (namespace, doc_id, "stub", "[]", 0.0,
             None, level or "L2", source),
        )
        store._conn.commit()
    monkeypatch.setattr(store, "upsert_vector", _stub_upsert)

    yield store
    store.close()


def test_precompact_snapshot_writes_session_log_vector(isolated_store, monkeypatch):
    """PreCompact CLI verb persists a marker into session:log namespace."""
    # Pre-populate session_log so the snapshot has tool-chain content.
    for tool in ("Read", "Grep", "Edit"):
        isolated_store.log_session_tool(
            session_id="s-1", query="x",
            query_vector=None, tool_used=tool, plugin_id=None,
        )

    # Patch SkillStore so the CLI verb uses our isolated store.
    from skill_hub import cli
    monkeypatch.setattr(cli, "SkillStore", lambda *a, **kw: isolated_store)
    # Prevent close() from invalidating the fixture.
    monkeypatch.setattr(isolated_store, "close", lambda: None)

    result = cli._cmd_precompact_snapshot(
        session_id="s-1", trigger="manual", transcript_path="/tmp/t.jsonl")

    assert result["decision"] == "allow"
    rows = isolated_store._conn.execute(
        "SELECT doc_id, namespace FROM vectors "
        "WHERE namespace = 'session:log' AND doc_id LIKE 'precompact:s-1:%'",
    ).fetchall()
    assert len(rows) == 1, f"expected one snapshot row, got {[dict(r) for r in rows]}"


def test_precompact_snapshot_no_session_id_is_noop(isolated_store, monkeypatch):
    from skill_hub import cli
    monkeypatch.setattr(cli, "SkillStore", lambda *a, **kw: isolated_store)
    monkeypatch.setattr(isolated_store, "close", lambda: None)

    result = cli._cmd_precompact_snapshot(session_id="")
    assert result == {"decision": "allow"}
    rows = isolated_store._conn.execute(
        "SELECT * FROM vectors WHERE namespace = 'session:log'",
    ).fetchall()
    assert rows == []


def test_session_close_writes_session_log_vector(isolated_store, monkeypatch):
    from skill_hub import cli
    monkeypatch.setattr(cli, "SkillStore", lambda *a, **kw: isolated_store)
    monkeypatch.setattr(isolated_store, "close", lambda: None)

    # Stub plugin_hooks.dispatch so we don't fan out to real plugins.
    from skill_hub import plugin_hooks
    dispatched: list[tuple] = []
    monkeypatch.setattr(plugin_hooks, "dispatch",
                        lambda evt, payload: dispatched.append((evt, payload)))

    result = cli._cmd_session_close(
        session_id="sess-99", reason="exit", summary="finished feature X")

    assert result == {"decision": "allow"}
    row = isolated_store._conn.execute(
        "SELECT doc_id, source FROM vectors "
        "WHERE namespace = 'session:log' AND doc_id = 'session:sess-99'",
    ).fetchone()
    assert row is not None
    assert row["source"] == "session_close"
    assert dispatched == [("on_session_end", {
        "session_id": "sess-99", "topic": "",
        "summary": "finished feature X", "reason": "exit",
    })]


def test_postcompact_optimize_returns_systemmessage(monkeypatch):
    """The CLI verb wraps optimize_memory output in a systemMessage payload."""
    from skill_hub import cli
    from skill_hub import config as _cfg_mod

    # Force dry-run to avoid mutating real memory files.
    monkeypatch.setattr(_cfg_mod, "get",
                        lambda k, default=None: False if k == "postcompact_optimize_apply"
                                                else default)
    # Stub the heavy server function so the test stays fast and offline.
    monkeypatch.setattr("skill_hub.server.optimize_memory",
                        lambda dry_run=True: f"REPORT (dry_run={dry_run})")

    result = cli._cmd_postcompact_optimize(session_id="s-1")
    assert result["decision"] == "allow"
    assert "REPORT (dry_run=True)" in result["systemMessage"]
    assert "PostCompact memory optimization" in result["systemMessage"]


def test_postcompact_optimize_truncates_huge_report(monkeypatch):
    from skill_hub import cli
    from skill_hub import config as _cfg_mod

    monkeypatch.setattr(_cfg_mod, "get",
                        lambda k, default=None: default)
    monkeypatch.setattr("skill_hub.server.optimize_memory",
                        lambda dry_run=True: "X" * 8000)

    result = cli._cmd_postcompact_optimize(session_id="s-1")
    assert "(truncated)" in result["systemMessage"]
    # 4000 chars + prefix + truncation marker, not the full 8000.
    assert len(result["systemMessage"]) < 5000
