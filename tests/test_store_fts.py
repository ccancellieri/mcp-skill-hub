"""Tests for FTS5 full-text search (BM25 fallback for tasks and teachings)."""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from skill_hub.store import SkillStore

    monkeypatch.setenv("HOME", str(tmp_path))
    return SkillStore(db_path=tmp_path / "skill_hub.db")


# ---------------------------------------------------------------------------
# Schema — FTS5 virtual tables exist
# ---------------------------------------------------------------------------

def test_fts_tables_created(store):
    """tasks_fts and teachings_fts virtual tables must exist after migration."""
    row = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='tasks_fts'"
    ).fetchone()
    assert row is not None, "tasks_fts table not found"

    row = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='teachings_fts'"
    ).fetchone()
    assert row is not None, "teachings_fts table not found"


def test_fts_triggers_created(store):
    """All six FTS5 sync triggers must exist."""
    expected = {
        "tasks_fts_insert",
        "tasks_fts_delete",
        "tasks_fts_update",
        "teachings_fts_insert",
        "teachings_fts_delete",
        "teachings_fts_update",
    }
    rows = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger'"
    ).fetchall()
    found = {r["name"] for r in rows}
    for trigger in expected:
        assert trigger in found, f"Missing trigger: {trigger}"


# ---------------------------------------------------------------------------
# search_text() — tasks
# ---------------------------------------------------------------------------

def test_search_text_returns_matching_task(store):
    """search_text('geoid') must find a task whose title contains 'geoid'."""
    store._conn.execute(
        "INSERT INTO tasks (title, summary, status) VALUES (?, ?, 'open')",
        ("geoid implementation", "Work on the geoid module"),
    )
    store._conn.commit()
    # FTS5 triggers fire on INSERT — index should be current.
    results = store.search_text("geoid", tables=["tasks"])
    assert len(results) >= 1
    assert results[0]["type"] == "tasks"
    assert "geoid" in results[0]["title_or_rule"].lower()


def test_search_text_no_match_returns_empty(store):
    """search_text with a query that matches nothing must return []."""
    store._conn.execute(
        "INSERT INTO tasks (title, summary, status) VALUES (?, ?, 'open')",
        ("python logging", "configure log levels"),
    )
    store._conn.commit()
    results = store.search_text("xyzzy_completely_unmatched", tables=["tasks"])
    assert results == []


def test_search_text_status_filter_open(store):
    """search_text with status='open' must only return open tasks."""
    store._conn.execute(
        "INSERT INTO tasks (title, summary, status) VALUES (?, ?, 'open')",
        ("open geoid task", "open task summary"),
    )
    store._conn.execute(
        "INSERT INTO tasks (title, summary, status, compact) "
        "VALUES (?, ?, 'closed', ?)",
        ("closed geoid task", "closed task summary", "compact text"),
    )
    store._conn.commit()

    results = store.search_text("geoid", tables=["tasks"], status="open")
    assert all(r["status"] == "open" for r in results)
    titles = [r["title_or_rule"] for r in results]
    assert any("open" in t for t in titles)


def test_search_text_top_k_respected(store):
    """search_text must not return more than top_k results."""
    for i in range(10):
        store._conn.execute(
            "INSERT INTO tasks (title, summary, status) VALUES (?, ?, 'open')",
            (f"geoid task {i}", f"summary about geoid number {i}"),
        )
    store._conn.commit()

    results = store.search_text("geoid", tables=["tasks"], top_k=3)
    assert len(results) <= 3


def test_search_text_result_keys(store):
    """Each result dict must contain the expected keys."""
    store._conn.execute(
        "INSERT INTO tasks (title, summary, status) VALUES (?, ?, 'open')",
        ("dynastore caching", "cache layer for dynastore"),
    )
    store._conn.commit()

    results = store.search_text("dynastore", tables=["tasks"])
    assert len(results) >= 1
    r = results[0]
    for key in ("id", "type", "title_or_rule", "summary_or_why", "score", "status"):
        assert key in r, f"Missing key '{key}' in result"


# ---------------------------------------------------------------------------
# search_text() — teachings
# ---------------------------------------------------------------------------

def test_search_text_teachings(store):
    """search_text must search teachings via FTS5."""
    store._conn.execute(
        "INSERT INTO teachings (rule, rule_vector, action, target_type, target_id) "
        "VALUES (?, ?, ?, ?, ?)",
        ("when I give a GitHub URL", "[]", "suggest chrome-devtools-mcp",
         "plugin", "chrome-devtools-mcp"),
    )
    store._conn.commit()

    results = store.search_text("GitHub URL", tables=["teachings"])
    assert len(results) >= 1
    assert results[0]["type"] == "teachings"
    assert "github" in results[0]["title_or_rule"].lower()


def test_search_text_all_tables(store):
    """When tables=None, both tasks and teachings are searched."""
    store._conn.execute(
        "INSERT INTO tasks (title, summary, status) VALUES (?, ?, 'open')",
        ("geoid task", "a task about geoid"),
    )
    store._conn.execute(
        "INSERT INTO teachings (rule, rule_vector, action, target_type, target_id) "
        "VALUES (?, ?, ?, ?, ?)",
        ("when working on geoid module", "[]", "use dynastore",
         "skill", "dynastore"),
    )
    store._conn.commit()

    results = store.search_text("geoid")  # tables=None → all
    types = {r["type"] for r in results}
    assert "tasks" in types
    assert "teachings" in types


# ---------------------------------------------------------------------------
# FTS5 triggers — insert/delete/update keep index in sync
# ---------------------------------------------------------------------------

def test_fts_trigger_insert_syncs(store):
    """Inserting a task populates FTS5 index via trigger."""
    cur = store._conn.execute(
        "INSERT INTO tasks (title, summary, status) VALUES (?, ?, 'open')",
        ("bandit router docs", "document route_to_model bandit"),
    )
    store._conn.commit()
    task_id = cur.lastrowid

    rows = store._conn.execute(
        "SELECT rowid FROM tasks_fts WHERE tasks_fts MATCH ?",
        ("bandit",),
    ).fetchall()
    ids = [r[0] for r in rows]
    assert task_id in ids


def test_fts_trigger_delete_syncs(store):
    """Deleting a task removes it from the FTS5 index."""
    cur = store._conn.execute(
        "INSERT INTO tasks (title, summary, status) VALUES (?, ?, 'open')",
        ("temporary geoid task", "this will be deleted"),
    )
    store._conn.commit()
    task_id = cur.lastrowid

    # Confirm indexed
    rows_before = store._conn.execute(
        "SELECT rowid FROM tasks_fts WHERE tasks_fts MATCH ?",
        ("temporary",),
    ).fetchall()
    assert any(r[0] == task_id for r in rows_before)

    # Delete and verify removal
    store._conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    store._conn.commit()

    rows_after = store._conn.execute(
        "SELECT rowid FROM tasks_fts WHERE tasks_fts MATCH ?",
        ("temporary",),
    ).fetchall()
    assert not any(r[0] == task_id for r in rows_after)


def test_fts_trigger_update_syncs(store):
    """Updating a task title keeps the FTS5 index in sync."""
    cur = store._conn.execute(
        "INSERT INTO tasks (title, summary, status) VALUES (?, ?, 'open')",
        ("old_title_xyz", "a task with unique title"),
    )
    store._conn.commit()
    task_id = cur.lastrowid

    # Update the title
    store._conn.execute(
        "UPDATE tasks SET title = ? WHERE id = ?",
        ("new_title_abc", task_id),
    )
    store._conn.commit()

    # Old title must not match
    rows_old = store._conn.execute(
        "SELECT rowid FROM tasks_fts WHERE tasks_fts MATCH ?",
        ("old_title_xyz",),
    ).fetchall()
    assert not any(r[0] == task_id for r in rows_old)

    # New title must match
    rows_new = store._conn.execute(
        "SELECT rowid FROM tasks_fts WHERE tasks_fts MATCH ?",
        ("new_title_abc",),
    ).fetchall()
    assert any(r[0] == task_id for r in rows_new)


# ---------------------------------------------------------------------------
# search_tasks() — FTS5 fallback when no vector
# ---------------------------------------------------------------------------

def test_search_tasks_fts_fallback(store):
    """search_tasks(query_vector=None, text_query='geoid') must use FTS5."""
    store._conn.execute(
        "INSERT INTO tasks (title, summary, status) VALUES (?, ?, 'open')",
        ("geoid pipeline", "end-to-end geoid ingestion pipeline"),
    )
    store._conn.commit()

    results = store.search_tasks(query_vector=None, text_query="geoid")
    assert len(results) >= 1
    assert results[0]["title_or_rule"] == "geoid pipeline"


def test_search_tasks_no_vector_no_text_returns_empty(store):
    """search_tasks with no vector and no text_query returns []."""
    store._conn.execute(
        "INSERT INTO tasks (title, summary, status) VALUES (?, ?, 'open')",
        ("some task", "some summary"),
    )
    store._conn.commit()

    results = store.search_tasks(query_vector=None, text_query=None)
    assert results == []


def test_search_tasks_fts_fallback_status_filter(store):
    """search_tasks FTS fallback respects the status filter."""
    store._conn.execute(
        "INSERT INTO tasks (title, summary, status) VALUES (?, ?, 'open')",
        ("open geoid work", "open summary"),
    )
    store._conn.execute(
        "INSERT INTO tasks (title, summary, status, compact) VALUES (?, ?, 'closed', ?)",
        ("closed geoid work", "closed summary", "compact"),
    )
    store._conn.commit()

    results = store.search_tasks(query_vector=None, text_query="geoid", status="open")
    assert all(r["status"] == "open" for r in results)


# ---------------------------------------------------------------------------
# search_text() — edge cases and graceful degradation
# ---------------------------------------------------------------------------

def test_search_text_empty_query_returns_empty(store):
    """Empty or whitespace-only query must return [] without error."""
    assert store.search_text("") == []
    assert store.search_text("   ") == []


def test_search_text_special_chars_do_not_raise(store):
    """FTS5 special characters in query must be sanitized, not raise."""
    store._conn.execute(
        "INSERT INTO tasks (title, summary, status) VALUES (?, ?, 'open')",
        ("test task", "a test summary"),
    )
    store._conn.commit()

    # These should not raise; results may or may not be found
    try:
        store.search_text('he said "hello"')
        store.search_text("(complex) -query *test")
        store.search_text("he said -- double dash")
    except Exception as exc:
        pytest.fail(f"search_text raised with special chars: {exc}")


def test_rebuild_fts_index_is_idempotent(store):
    """Calling _rebuild_fts_index twice must not raise or corrupt data."""
    store._conn.execute(
        "INSERT INTO tasks (title, summary, status) VALUES (?, ?, 'open')",
        ("geoid rebuild test", "testing idempotent rebuild"),
    )
    store._conn.commit()

    # Call rebuild twice — must not raise
    store._rebuild_fts_index(store._conn)
    store._rebuild_fts_index(store._conn)

    results = store.search_text("rebuild", tables=["tasks"])
    assert len(results) >= 1
