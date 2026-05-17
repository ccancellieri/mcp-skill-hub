"""Tests for M4-3 federation-lite: WAL mode + node_id + federation_view.

Covers issue #22 acceptance:
- Two-DB attach + cross-host queries correct.
- WAL migration idempotent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SKILL_HUB_NODE_ID", "host-alpha")
    from skill_hub.store import SkillStore

    return SkillStore(db_path=tmp_path / "alpha.db")


@pytest.fixture()
def remote_store(tmp_path, monkeypatch):
    """A second store impersonating a different host on disk."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SKILL_HUB_NODE_ID", "host-beta")
    from skill_hub.store import SkillStore

    return SkillStore(db_path=tmp_path / "beta.db")


# ---------------------------------------------------------------------------
# WAL — idempotent migration
# ---------------------------------------------------------------------------


def test_wal_mode_enabled(store):
    """SQLite must be running in WAL after init."""
    mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_wal_migration_idempotent(tmp_path, monkeypatch):
    """Re-opening the same DB must keep journal_mode = wal and not error."""
    monkeypatch.setenv("HOME", str(tmp_path))
    from skill_hub.store import SkillStore

    db = tmp_path / "x.db"
    s1 = SkillStore(db_path=db)
    assert s1._conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    s1._conn.close()

    s2 = SkillStore(db_path=db)
    assert s2._conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    # Re-running the explicit PRAGMA again must still succeed.
    assert s2._conn.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() == "wal"


# ---------------------------------------------------------------------------
# node_id — resolution + propagation
# ---------------------------------------------------------------------------


def test_node_id_resolves_from_env(store):
    """SKILL_HUB_NODE_ID env var wins when config is silent."""
    assert store.node_id == "host-alpha"


def test_node_id_sanitised(tmp_path, monkeypatch):
    """Disallowed characters in node_id are replaced — keeps SQL identifiers safe."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SKILL_HUB_NODE_ID", "weird name/with spaces!")
    from skill_hub.store import SkillStore

    s = SkillStore(db_path=tmp_path / "n.db")
    # Replaced run becomes a single underscore; trailing punctuation stripped.
    assert s.node_id == "weird_name_with_spaces"


def test_tasks_table_has_node_id_column(store):
    cols = {row[1] for row in store._conn.execute("PRAGMA table_info(tasks)")}
    assert "node_id" in cols


def test_save_task_stamps_node_id(store):
    task_id = store.save_task(
        title="alpha task",
        summary="some summary",
        vector=[0.0] * 16,
    )
    row = store._conn.execute(
        "SELECT node_id FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    assert row["node_id"] == "host-alpha"


# ---------------------------------------------------------------------------
# events table — schema from M2 design + node_id
# ---------------------------------------------------------------------------


def test_events_table_exists(store):
    row = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='events'"
    ).fetchone()
    assert row is not None


def test_events_table_columns(store):
    cols = {row[1] for row in store._conn.execute("PRAGMA table_info(events)")}
    for required in ("id", "session_id", "ts", "kind", "tool_name", "payload", "node_id"):
        assert required in cols, f"events.{required} missing"


def test_append_event_stamps_node_id(store):
    eid = store.append_event(
        session_id="sess-1",
        kind="tool_invoke",
        tool_name="search_skills",
        payload={"q": "hello"},
    )
    assert eid > 0
    row = store._conn.execute(
        "SELECT node_id, kind, payload FROM events WHERE id = ?", (eid,)
    ).fetchone()
    assert row["node_id"] == "host-alpha"
    assert row["kind"] == "tool_invoke"
    assert '"q": "hello"' in row["payload"]


# ---------------------------------------------------------------------------
# federation_view — two-DB ATTACH + cross-host queries
# ---------------------------------------------------------------------------


def test_federation_view_missing_path_raises(store, tmp_path):
    with pytest.raises(FileNotFoundError):
        store.federation_view(tmp_path / "nope.db")


def test_federation_view_two_db_attach_correct(store, remote_store):
    """The acceptance test: attach a remote DB, see its node_id + counts."""
    # Seed local + remote with one task each.
    store.save_task(title="alpha-local", summary="local task", vector=[0.0] * 16)
    remote_store.save_task(title="beta-remote", summary="remote task", vector=[0.0] * 16)
    remote_store.append_event(
        session_id="sess-r",
        kind="tool_invoke",
        tool_name="search_skills",
        payload={"q": "hi"},
    )
    # Force flush to disk on the remote so the read-only attach sees it.
    remote_store._conn.commit()

    info = store.federation_view(remote_store.db_path)

    assert info["local_node"] == "host-alpha"
    assert "host-beta" in info["remote_nodes"]
    assert info["tasks"]["local"] == 1
    assert info["tasks"]["remote"] == 1
    assert info["events"]["remote"] == 1
    assert info["schemas"]["tasks_remote"] is True
    assert info["schemas"]["events_remote"] is True

    # The remote DB must be detached again (no leaking schema).
    schemas = [
        row[1] for row in store._conn.execute("PRAGMA database_list").fetchall()
    ]
    assert "remote" not in schemas


def test_federation_view_alias_sanitised(store, remote_store, tmp_path):
    """Bad alias characters must not blow up the ATTACH."""
    info = store.federation_view(remote_store.db_path, alias="weird name!")
    assert info["alias"] == "weird_name"


def test_federation_view_against_empty_db(tmp_path, store):
    """Pointing at a non-skill-hub DB must report empty cleanly, not crash."""
    import sqlite3

    junk = tmp_path / "junk.db"
    conn = sqlite3.connect(str(junk))
    conn.execute("CREATE TABLE other (x INTEGER)")
    conn.commit()
    conn.close()

    info = store.federation_view(junk)
    assert info["tasks"]["remote"] == 0
    assert info["events"]["remote"] == 0
    assert info["schemas"]["tasks_remote"] is False
    assert info["schemas"]["events_remote"] is False
    assert info["remote_nodes"] == []
